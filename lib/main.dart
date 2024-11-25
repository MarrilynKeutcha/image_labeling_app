import 'dart:io';
import 'dart:typed_data';
import 'package:flutter/material.dart';
import 'package:image_picker/image_picker.dart';
import 'package:tflite_flutter/tflite_flutter.dart';
import 'package:image/image.dart' as img; // For image processing
import 'package:flutter/services.dart' show rootBundle; // For loading assets

void main() {
  runApp(MyApp());
}

class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Image Labeling App',
      theme: ThemeData(
        primarySwatch: Colors.blue,
      ),
      home: ImageLabelingScreen(),
    );
  }
}

class ImageLabelingScreen extends StatefulWidget {
  @override
  _ImageLabelingScreenState createState() => _ImageLabelingScreenState();
}

class _ImageLabelingScreenState extends State<ImageLabelingScreen> {
  File? _image;
  List<Map<String, dynamic>> _labels = [];
  final ImagePicker _picker = ImagePicker();
  late Interpreter _interpreter;

  @override
  void initState() {
    super.initState();
    _loadModel();
  }

  // Load the TensorFlow Lite model
  Future<void> _loadModel() async {
    try {
      _interpreter = await Interpreter.fromAsset('assets/model.tflite');
      print("Model loaded successfully!");
    } catch (e) {
      print("Error loading model: $e");
    }
  }

  // Pick an image from camera or gallery
  Future<void> _pickImage(ImageSource source) async {
    final pickedFile = await _picker.pickImage(source: source);
    if (pickedFile != null) {
      final imageBytes = await pickedFile.readAsBytes();
      setState(() {
        _image = File(pickedFile.path);
        _labels.clear(); // Clear previous results
      });

      // Run image labeling
      _labelImage(imageBytes);
    }
  }

  // Label the image using TensorFlow Lite
  Future<void> _labelImage(Uint8List imageBytes) async {
    try {
      // Preprocess the image
      final input = _preprocessImage(imageBytes);

      // Define the output buffer (e.g., for MobileNet, 1001 classes)
      final output = List.filled(1 * 1001, 0.0).reshape([1, 1001]);

      // Run inference
      _interpreter.run(input, output);

      // Postprocess the output
      final predictions = await _postprocessOutput(output);
      setState(() {
        _labels = predictions;
      });

      print("Image labeled successfully!");
    } catch (e) {
      print("Error during inference: $e");
    }
  }

  // Preprocess the image to match model input format
  List<List<List<List<double>>>> _preprocessImage(Uint8List imageBytes) {
    // Decode the image
    final image = img.decodeImage(imageBytes);

    // Resize the image to 224x224 (example size for MobileNet)
    final resizedImage = img.copyResize(image!, width: 224, height: 224);

    // Normalize pixel values to [0, 1]
    final input = List.generate(1, (batch) {
      return List.generate(224, (y) {
        return List.generate(224, (x) {
          final pixel = resizedImage.getPixel(x, y);
          final r = img.getRed(pixel) / 255.0;
          final g = img.getGreen(pixel) / 255.0;
          final b = img.getBlue(pixel) / 255.0;
          return [r, g, b]; // Return normalized RGB values
        });
      });
    });

    return input;
  }

  // Map model output to meaningful labels
  Future<List<Map<String, dynamic>>> _postprocessOutput(List<List<double>> output) async {
    // Load the labels from the assets folder
    final labels = await rootBundle.loadString('assets/labels.txt').then((value) => value.split('\n'));

    // Find the top predictions
    final predictions = List.generate(output[0].length, (i) {
      return {"label": labels[i], "confidence": (output[0][i] * 100).toStringAsFixed(2)};
    });

    // Sort predictions by confidence
    predictions.sort((a, b) => double.parse(b["confidence"]).compareTo(double.parse(a["confidence"])));

    // Return the top 5 predictions
    return predictions.take(5).toList();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: Text("Image Labeling App"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            // Display the selected image
            _image == null
                ? Text(
                    "No image selected.",
                    style: TextStyle(fontSize: 18),
                  )
                : Image.file(
                    _image!,
                    height: 300,
                  ),
            SizedBox(height: 16),

            // Buttons to pick an image
            Row(
              mainAxisAlignment: MainAxisAlignment.spaceEvenly,
              children: [
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.camera),
                  icon: Icon(Icons.camera_alt),
                  label: Text("Camera"),
                ),
                ElevatedButton.icon(
                  onPressed: () => _pickImage(ImageSource.gallery),
                  icon: Icon(Icons.photo_library),
                  label: Text("Gallery"),
                ),
              ],
            ),
            SizedBox(height: 16),

            // Display detected labels
            Expanded(
              child: _labels.isEmpty
                  ? Text(
                      "No labels detected yet.",
                      style: TextStyle(fontSize: 16),
                    )
                  : ListView.builder(
                      itemCount: _labels.length,
                      itemBuilder: (context, index) {
                        final label = _labels[index];
                        return ListTile(
                          title: Text(
                            label["label"],
                            style: TextStyle(fontSize: 18),
                          ),
                          subtitle: Text(
                            "Confidence: ${label["confidence"]}%",
                            style: TextStyle(fontSize: 14, color: Colors.grey),
                          ),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
    );
  }
}
