import 'dart:io';
import 'dart:math' as math;
import 'dart:collection';
import 'package:camera/camera.dart';
import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:google_mlkit_pose_detection/google_mlkit_pose_detection.dart';
import 'package:permission_handler/permission_handler.dart';

late List<CameraDescription> _cameras;

Future<void> main() async {
  WidgetsFlutterBinding.ensureInitialized();
  try {
    _cameras = await availableCameras();
  } catch (e) {
    debugPrint("Error initializing cameras: $e");
    _cameras = [];
  }
  runApp(const MaterialApp(home: PushupCounterApp()));
}

/// Smooths angle readings to reduce noise
class AngleSmoother {
  final int windowSize;
  final Queue<double> _values = Queue();

  AngleSmoother({this.windowSize = 5});

  double add(double value) {
    _values.add(value);
    if (_values.length > windowSize) {
      _values.removeFirst();
    }
    return _values.reduce((a, b) => a + b) / _values.length;
  }

  void reset() => _values.clear();
  bool get hasEnoughData => _values.length >= windowSize;
}

/// Tracks position history for motion validation
class PositionTracker {
  final int historySize;
  final Queue<double> _history = Queue();

  PositionTracker({this.historySize = 10});

  void add(double value) {
    _history.add(value);
    if (_history.length > historySize) {
      _history.removeFirst();
    }
  }

  double? get trend {
    if (_history.length < 3) return null;
    final list = _history.toList();
    return list.last - list.first;
  }

  double? get range {
    if (_history.isEmpty) return null;
    final list = _history.toList();
    return list.reduce(math.max) - list.reduce(math.min);
  }

  void reset() => _history.clear();
}

enum PushupStage {
  notInPosition, // User not in valid pushup stance
  up,            // Arms extended (top position)
  goingDown,     // Transitioning to bottom
  down,          // Arms bent (bottom position)
  goingUp,       // Transitioning to top
}

class PushupCounterApp extends StatefulWidget {
  const PushupCounterApp({super.key});

  @override
  State<PushupCounterApp> createState() => _PushupCounterAppState();
}

class _PushupCounterAppState extends State<PushupCounterApp> {
  CameraController? _controller;
  late final PoseDetector _poseDetector;
  bool _isBusy = false;

  // ==================== STATE ====================
  int _counter = 0;
  PushupStage _stage = PushupStage.notInPosition;
  double _currentElbowAngle = 0.0;
  double _bodyAngle = 0.0;
  double _armSymmetry = 0.0;
  String _feedback = "Get into pushup position";
  Color _statusColor = Colors.grey;

  // ==================== SMOOTHING & TRACKING ====================
  final AngleSmoother _elbowSmoother = AngleSmoother(windowSize: 5);
  final AngleSmoother _bodySmoother = AngleSmoother(windowSize: 5);
  final PositionTracker _elbowTracker = PositionTracker(historySize: 15);

  // ==================== THRESHOLDS ====================
  // Elbow angle thresholds
  // 160 is a good lockout, 125 is deep enough to count as a rep without forcing chest-to-floor
  static const double _elbowUpThreshold = 160.0;
  static const double _elbowDownThreshold = 125.0;

  // Body alignment (shoulder-hip-ankle)
  // Lowered to 135 to allow for some sagging or camera angle distortion
  static const double _minBodyAlignment = 135.0;

  // Arm symmetry (both arms should move together)
  // Increased tolerance to 45 degrees
  static const double _maxArmAsymmetry = 45.0;

  // Landmark confidence threshold
  static const double _minConfidence = 0.5;

  // Consecutive frames required to confirm position
  static const int _framesRequired = 3; // Reduced to 3 for snappier response

  // Minimum time between reps (prevents double counting)
  static const Duration _minRepDuration = Duration(milliseconds: 600);

  // ==================== COUNTERS ====================
  int _framesInUp = 0;
  int _framesInDown = 0;
  DateTime? _lastRepTime;
  bool _reachedBottom = false;

  @override
  void initState() {
    super.initState();
    _poseDetector = PoseDetector(
      options: PoseDetectorOptions(
        mode: PoseDetectionMode.stream,
        model: PoseDetectionModel.accurate,
      ),
    );
    _initializeCamera();
  }

  Future<void> _initializeCamera() async {
    if (_cameras.isEmpty) return;
    
    final status = await Permission.camera.request();
    if (status.isDenied) return;

    final camera = _cameras.firstWhere(
      (c) => c.lensDirection == CameraLensDirection.front,
      orElse: () => _cameras.first,
    );

    _controller = CameraController(
      camera,
      ResolutionPreset.medium,
      enableAudio: false,
      imageFormatGroup:
          Platform.isAndroid ? ImageFormatGroup.nv21 : ImageFormatGroup.bgra8888,
    );

    await _controller!.initialize();
    await _controller!.startImageStream(_processImage);

    if (mounted) setState(() {});
  }

  Future<void> _processImage(CameraImage image) async {
    if (_isBusy) return;
    _isBusy = true;

    try {
      final inputImage = _inputImageFromCameraImage(image);
      if (inputImage == null) {
        _isBusy = false;
        return;
      }

      final poses = await _poseDetector.processImage(inputImage);

      if (poses.isNotEmpty) {
        _analyzePose(poses.first);
      } else {
        _setNotInPosition("Body not detected");
      }
    } catch (e) {
      debugPrint("Error processing image: $e");
    } finally {
      _isBusy = false;
    }
  }

  void _analyzePose(Pose pose) {
    // 1. EXTRACT & VALIDATE LANDMARKS
    // We are now lenient on legs/ankles.
    final validation = _validateLandmarks(pose);
    if (!validation.isValid) {
      _setNotInPosition(validation.message);
      return;
    }

    final lm = validation.landmarks!;

    // 2. CALCULATE ANGLES
    // Body Alignment
    final bodyAngle = _calculateBodyAlignment(lm);
    final smoothedBodyAngle = _bodySmoother.add(bodyAngle);

    // Elbow Angles
    final leftElbow = _calculateAngle(
      lm[PoseLandmarkType.leftShoulder]!,
      lm[PoseLandmarkType.leftElbow]!,
      lm[PoseLandmarkType.leftWrist]!,
    );
    final rightElbow = _calculateAngle(
      lm[PoseLandmarkType.rightShoulder]!,
      lm[PoseLandmarkType.rightElbow]!,
      lm[PoseLandmarkType.rightWrist]!,
    );
    
    final avgElbowAngle = (leftElbow + rightElbow) / 2;
    final smoothedElbow = _elbowSmoother.add(avgElbowAngle);
    _elbowTracker.add(smoothedElbow);
    
    final armDiff = (leftElbow - rightElbow).abs();

    // 3. GENERATE WARNINGS (BUT DO NOT RESET PROGRESS)
    // Instead of killing the rep with _setNotInPosition, we just generate a warning string.
    String? warningMessage;
    
    if (smoothedBodyAngle < _minBodyAlignment) {
      warningMessage = "Straighten Back!";
    } else if (armDiff > _maxArmAsymmetry) {
      warningMessage = "Fix Balance!";
    } else if (!_validateHandPosition(lm)) {
      warningMessage = "Hands Under Shoulders";
    }

    // 4. UPDATE STATE MACHINE
    // We pass the warning to the state machine. It will change the text color,
    // but it won't reset the counter.
    _updateStateMachine(smoothedElbow, smoothedBodyAngle, armDiff, warning: warningMessage);
  }

  /// Validates that required landmarks are present.
  /// We are STRICT on upper body, but LENIENT on lower body (ankles).
  _LandmarkValidation _validateLandmarks(Pose pose) {
    final requiredUpperBody = [
      PoseLandmarkType.leftShoulder,
      PoseLandmarkType.rightShoulder,
      PoseLandmarkType.leftElbow,
      PoseLandmarkType.rightElbow,
      PoseLandmarkType.leftWrist,
      PoseLandmarkType.rightWrist,
    ];
    
    // We need hips to check form, but knees/ankles are optional if hips are visible
    final requiredHips = [
      PoseLandmarkType.leftHip,
      PoseLandmarkType.rightHip,
    ];

    final landmarks = <PoseLandmarkType, PoseLandmark>{};
    
    // 1. Check Upper Body (Crucial)
    for (final type in requiredUpperBody) {
      final lm = pose.landmarks[type];
      if (lm == null || lm.likelihood < _minConfidence) {
        return _LandmarkValidation(isValid: false, message: "Can't see upper body");
      }
      landmarks[type] = lm;
    }

    // 2. Check Hips (Crucial for body alignment)
    for (final type in requiredHips) {
      final lm = pose.landmarks[type];
      if (lm == null || lm.likelihood < _minConfidence) {
        return _LandmarkValidation(isValid: false, message: "Can't see hips");
      }
      landmarks[type] = lm;
    }

    // 3. Try to get legs (Optional - for better accuracy, but don't fail without them)
    final legTypes = [
      PoseLandmarkType.leftKnee, PoseLandmarkType.rightKnee,
      PoseLandmarkType.leftAnkle, PoseLandmarkType.rightAnkle
    ];
    
    for (final type in legTypes) {
      final lm = pose.landmarks[type];
      if (lm != null && lm.likelihood >= _minConfidence) {
        landmarks[type] = lm;
      }
    }

    return _LandmarkValidation(isValid: true, landmarks: landmarks);
  }

  String _landmarkName(PoseLandmarkType type) {
    return type.name.replaceAll(RegExp(r'([A-Z])'), ' \$1').trim().toLowerCase();
  }

  /// Checks if body is in a straight plank position
  double _calculateBodyAlignment(Map<PoseLandmarkType, PoseLandmark> lm) {
    // Helper to get angle safely even if ankles/knees are missing
    double getSideAngle(PoseLandmarkType shoulder, PoseLandmarkType hip, PoseLandmarkType knee, PoseLandmarkType ankle) {
      if (lm.containsKey(ankle) && lm.containsKey(hip) && lm.containsKey(shoulder)) {
        return _calculateAngle(lm[shoulder]!, lm[hip]!, lm[ankle]!);
      } else if (lm.containsKey(knee) && lm.containsKey(hip) && lm.containsKey(shoulder)) {
        // Fallback to Knee
        return _calculateAngle(lm[shoulder]!, lm[hip]!, lm[knee]!);
      }
      return 180.0; // Default to perfect alignment if we can't see legs
    }

    final leftAngle = getSideAngle(
      PoseLandmarkType.leftShoulder, PoseLandmarkType.leftHip, 
      PoseLandmarkType.leftKnee, PoseLandmarkType.leftAnkle
    );
    
    final rightAngle = getSideAngle(
      PoseLandmarkType.rightShoulder, PoseLandmarkType.rightHip, 
      PoseLandmarkType.rightKnee, PoseLandmarkType.rightAnkle
    );

    return (leftAngle + rightAngle) / 2;
  }

  /// Checks if hands are properly positioned under/near shoulders
  bool _validateHandPosition(Map<PoseLandmarkType, PoseLandmark> lm) {
    final leftShoulder = lm[PoseLandmarkType.leftShoulder]!;
    final rightShoulder = lm[PoseLandmarkType.rightShoulder]!;
    final leftWrist = lm[PoseLandmarkType.leftWrist]!;
    final rightWrist = lm[PoseLandmarkType.rightWrist]!;

    final shoulderY = (leftShoulder.y + rightShoulder.y) / 2;
    final wristY = (leftWrist.y + rightWrist.y) / 2;

    return wristY > shoulderY - 100; // Allow 100 pixels of tolerance
  }

  void _updateStateMachine(double elbowAngle, double bodyAngle, double armSymmetry, {String? warning}) {
    final isUp = elbowAngle >= _elbowUpThreshold;
    final isDown = elbowAngle <= _elbowDownThreshold;

    // Update frame counters
    if (isUp) {
      _framesInUp++;
      _framesInDown = 0;
    } else if (isDown) {
      _framesInDown++;
      _framesInUp = 0;
    } else {
      // In transition - decay counters slowly
      _framesInUp = math.max(0, _framesInUp - 1);
      _framesInDown = math.max(0, _framesInDown - 1);
    }

    PushupStage newStage = _stage;
    String newFeedback = _feedback;
    Color newColor = _statusColor;
    
    // If a warning exists, we prioritize showing it, but we continue processing stages
    bool hasWarning = warning != null;

    switch (_stage) {
      case PushupStage.notInPosition:
        // Just entered valid position
        if (isUp && _framesInUp >= _framesRequired) {
          newStage = PushupStage.up;
          newFeedback = "Good! Now go down";
          newColor = Colors.green;
          _reachedBottom = false;
        } else if (isDown && _framesInDown >= _framesRequired) {
          newStage = PushupStage.down;
          newFeedback = "Push up!";
          newColor = Colors.orange;
          _reachedBottom = true;
        } else {
          newFeedback = "Get into start position";
          newColor = Colors.blue;
        }
        break;

      case PushupStage.up:
        if (isDown && _framesInDown >= _framesRequired) {
          newStage = PushupStage.down;
          newFeedback = "Good depth! Push up!";
          newColor = Colors.orange;
          _reachedBottom = true;
        } else if (!isUp) {
          newStage = PushupStage.goingDown;
          newFeedback = "Going down...";
          newColor = Colors.yellow;
        }
        break;

      case PushupStage.goingDown:
        if (isDown && _framesInDown >= _framesRequired) {
          newStage = PushupStage.down;
          newFeedback = "Great! Now push up!";
          newColor = Colors.orange;
          _reachedBottom = true;
        } else if (isUp && _framesInUp >= _framesRequired) {
          // Went back up without reaching bottom
          newStage = PushupStage.up;
          newFeedback = "Go lower next time!";
          newColor = Colors.green;
        }
        break;

      case PushupStage.down:
        if (isUp && _framesInUp >= _framesRequired) {
          // Completed a rep!
          if (_reachedBottom && _canCountRep()) {
            _counter++;
            _lastRepTime = DateTime.now();
          }
          newStage = PushupStage.up;
          newFeedback = "✓ ${_counter} - Go again!";
          newColor = Colors.green;
          _reachedBottom = false;
        } else if (!isDown) {
          newStage = PushupStage.goingUp;
          newFeedback = "Push! Push!";
          newColor = Colors.yellow;
        }
        break;

      case PushupStage.goingUp:
        if (isUp && _framesInUp >= _framesRequired) {
          // Completed a rep!
          if (_reachedBottom && _canCountRep()) {
            _counter++;
            _lastRepTime = DateTime.now();
          }
          newStage = PushupStage.up;
          newFeedback = "✓ ${_counter} - Great form!";
          newColor = Colors.green;
          _reachedBottom = false;
        } else if (isDown && _framesInDown >= _framesRequired) {
          newStage = PushupStage.down;
          newFeedback = "Full extension up!";
          newColor = Colors.orange;
        }
        break;
    }

    // Override feedback with warning if present, but keep the stage
    if (hasWarning && newStage != PushupStage.notInPosition) {
       newFeedback = warning;
       newColor = Colors.orange;
    }

    if (mounted) {
      setState(() {
        _stage = newStage;
        _feedback = newFeedback;
        _statusColor = newColor;
        _currentElbowAngle = elbowAngle;
        _bodyAngle = bodyAngle;
        _armSymmetry = armSymmetry;
      });
    }
  }

  bool _canCountRep() {
    if (_lastRepTime == null) return true;
    return DateTime.now().difference(_lastRepTime!) >= _minRepDuration;
  }

  // This is only called when we completely lose track of the user (e.g. no upper body seen)
  void _setNotInPosition(String message, {
    double? elbowAngle,
    double? bodyAngle,
    double? armSymmetry,
  }) {
    _framesInUp = 0;
    _framesInDown = 0;
    _reachedBottom = false;

    if (mounted) {
      setState(() {
        _stage = PushupStage.notInPosition;
        _feedback = message;
        _statusColor = Colors.red;
        if (elbowAngle != null) _currentElbowAngle = elbowAngle;
        if (bodyAngle != null) _bodyAngle = bodyAngle;
        if (armSymmetry != null) _armSymmetry = armSymmetry;
      });
    }
  }

  double _calculateAngle(PoseLandmark first, PoseLandmark mid, PoseLandmark last) {
    final radians = math.atan2(last.y - mid.y, last.x - mid.x) -
        math.atan2(first.y - mid.y, first.x - mid.x);
    var degrees = radians * 180.0 / math.pi;
    degrees = degrees.abs();
    if (degrees > 180.0) degrees = 360.0 - degrees;
    return degrees;
  }

  InputImage? _inputImageFromCameraImage(CameraImage image) {
    if (_controller == null) return null;

    final camera = _controller!.description;
    final sensorOrientation = camera.sensorOrientation;

    InputImageRotation? rotation;
    if (Platform.isIOS) {
      rotation = InputImageRotationValue.fromRawValue(sensorOrientation);
    } else if (Platform.isAndroid) {
      var rotationCompensation =
          _orientations[_controller!.value.deviceOrientation];
      if (rotationCompensation == null) return null;
      if (camera.lensDirection == CameraLensDirection.front) {
        rotationCompensation = (sensorOrientation + rotationCompensation) % 360;
      } else {
        rotationCompensation =
            (sensorOrientation - rotationCompensation + 360) % 360;
      }
      rotation = InputImageRotationValue.fromRawValue(rotationCompensation);
    }
    if (rotation == null) return null;

    final format = InputImageFormatValue.fromRawValue(image.format.raw);
    if (format == null ||
        (Platform.isAndroid && format != InputImageFormat.nv21)) {
      return null;
    }

    if (image.planes.isEmpty) return null;
    final plane = image.planes.first;

    return InputImage.fromBytes(
      bytes: plane.bytes,
      metadata: InputImageMetadata(
        size: Size(image.width.toDouble(), image.height.toDouble()),
        rotation: rotation,
        format: format,
        bytesPerRow: plane.bytesPerRow,
      ),
    );
  }

  final _orientations = {
    DeviceOrientation.portraitUp: 0,
    DeviceOrientation.landscapeLeft: 90,
    DeviceOrientation.portraitDown: 180,
    DeviceOrientation.landscapeRight: 270,
  };

  void _resetCounter() {
    setState(() {
      _counter = 0;
      _stage = PushupStage.notInPosition;
      _framesInUp = 0;
      _framesInDown = 0;
      _reachedBottom = false;
      _elbowSmoother.reset();
      _bodySmoother.reset();
      _elbowTracker.reset();
      _feedback = "Get into pushup position";
      _statusColor = Colors.grey;
    });
  }

  @override
  void dispose() {
    _controller?.dispose();
    _poseDetector.close();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_controller == null || !_controller!.value.isInitialized) {
      return const Scaffold(
        body: Center(child: CircularProgressIndicator()),
      );
    }

    return Scaffold(
      body: Stack(
        fit: StackFit.expand,
        children: [
          // Camera Preview
          CameraPreview(_controller!),

          // Main Overlay
          Positioned(
            top: 50,
            left: 16,
            right: 16,
            child: Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.7),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  // Counter
                  Text(
                    "$_counter",
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 72,
                      fontWeight: FontWeight.bold,
                    ),
                  ),

                  // Feedback banner
                  Container(
                    width: double.infinity,
                    padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                    decoration: BoxDecoration(
                      color: _statusColor.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(12),
                      border: Border.all(color: _statusColor, width: 2),
                    ),
                    child: Text(
                      _feedback,
                      textAlign: TextAlign.center,
                      style: TextStyle(
                        color: _statusColor,
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ),

                  const SizedBox(height: 16),

                  // Elbow angle indicator
                  _buildAngleIndicator(
                    label: "Elbow Angle",
                    value: _currentElbowAngle,
                    min: _elbowDownThreshold,
                    max: _elbowUpThreshold,
                    lowLabel: "DOWN",
                    highLabel: "UP",
                  ),

                  const SizedBox(height: 8),

                  // Body alignment indicator
                  _buildAngleIndicator(
                    label: "Body Alignment",
                    value: _bodyAngle,
                    min: 120,
                    max: 180,
                    lowLabel: "BENT",
                    highLabel: "STRAIGHT",
                    threshold: _minBodyAlignment,
                  ),
                ],
              ),
            ),
          ),

          // Debug panel (remove in production)
          Positioned(
            bottom: 100,
            left: 16,
            child: Container(
              padding: const EdgeInsets.all(12),
              decoration: BoxDecoration(
                color: Colors.black.withOpacity(0.7),
                borderRadius: BorderRadius.circular(8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text("Stage: ${_stage.name}",
                      style: const TextStyle(color: Colors.white, fontSize: 12)),
                  Text("Frames Up: $_framesInUp | Down: $_framesInDown",
                      style: const TextStyle(color: Colors.white70, fontSize: 11)),
                  Text("Arm Diff: ${_armSymmetry.toStringAsFixed(1)}°",
                      style: const TextStyle(color: Colors.white70, fontSize: 11)),
                ],
              ),
            ),
          ),

          // Reset button
          Positioned(
            bottom: 30,
            right: 30,
            child: FloatingActionButton(
              onPressed: _resetCounter,
              backgroundColor: Colors.red.shade700,
              child: const Icon(Icons.refresh),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAngleIndicator({
    required String label,
    required double value,
    required double min,
    required double max,
    required String lowLabel,
    required String highLabel,
    double? threshold,
  }) {
    final progress = ((value - min) / (max - min)).clamp(0.0, 1.0);
    final isAboveThreshold = threshold == null || value >= threshold;

    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(label, style: const TextStyle(color: Colors.white70, fontSize: 12)),
            Text(
              "${value.toStringAsFixed(0)}°",
              style: TextStyle(
                color: isAboveThreshold ? Colors.white : Colors.red,
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
          ],
        ),
        const SizedBox(height: 4),
        Stack(
          children: [
            LinearProgressIndicator(
              value: progress,
              backgroundColor: Colors.grey.shade800,
              color: isAboveThreshold ? _statusColor : Colors.red,
              minHeight: 8,
            ),
            if (threshold != null)
              Positioned(
                left: ((threshold - min) / (max - min)).clamp(0.0, 1.0) *
                    MediaQuery.of(context).size.width *
                    0.85, // approximate width
                child: Container(
                  width: 2,
                  height: 8,
                  color: Colors.white,
                ),
              ),
          ],
        ),
        const SizedBox(height: 2),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(lowLabel, style: const TextStyle(color: Colors.orange, fontSize: 10)),
            Text(highLabel, style: const TextStyle(color: Colors.green, fontSize: 10)),
          ],
        ),
      ],
    );
  }
}

class _LandmarkValidation {
  final bool isValid;
  final String message;
  final Map<PoseLandmarkType, PoseLandmark>? landmarks;

  _LandmarkValidation({
    required this.isValid,
    this.message = "",
    this.landmarks,
  });
}