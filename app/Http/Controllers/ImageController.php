<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Validator;
class ImageController extends Controller
{
//    public function processImage(Request $request)
//     {
//         $request->validate([
//             'image' => 'required|image|mimes:jpg,png,jpeg|max:2048',
//         ]);
        
//         $image = $request->file('image');
//         $imageName = time() . '.' . $image->extension();
        
//         // 🔹 Image ko public directory me store karein
//         $image->move(public_path('user/image'), $imageName);
        
//         // 🔹 Python script ke liye full image path generate karein
//         $imagePath = public_path('user/image/' . $imageName);
//         \Log::info("Stored Image Path: " . $imagePath);
        
//         $pythonPath = "C:/Users/pc/myenv/Scripts/python.exe"; // ✅ Virtualenv ka Python
//         $scriptPath = "C:/Users/pc/Desktop/bmi-calculator/bmi-backend-python/bmi-backend/scripts/extract_height_weight.py";
        
//         $command = escapeshellcmd("\"$pythonPath\" \"$scriptPath\" \"$imagePath\"");
//         $output = shell_exec($command);
        
//         \Log::info("Python script raw output: " . $output);
        
//         // 🔹 Multi-line output ka sirf last line parse karo
//         $lines = explode("\n", trim($output));
//         $jsonString = end($lines); // ✅ Sirf last JSON line lo
        
//         $jsonOutput = json_decode($jsonString, true);
        
//         if ($jsonOutput === null) {
//             return response()->json([
//                 'error' => 'Invalid JSON response from Python script',
//                 'raw_output' => $output // Debugging ke liye pura output bhejo
//             ], 500);
//         }

//         // ✅ Check if height and weight exist
//         if (!isset($jsonOutput['height']) || !isset($jsonOutput['weight'])) {
//             return response()->json([
//                 'error' => 'Missing height or weight in response',
//                 'raw_output' => $jsonOutput
//             ], 500);
//         }

//         // ✅ BMI Calculation
//         $height_m = $jsonOutput['height'] / 100; // cm to meters
//         $weight_kg = $jsonOutput['weight']; // kg
//         $bmi = round($weight_kg / ($height_m * $height_m), 2); // ✅ BMI Formula
    
//         // ✅ Final Response with BMI
//         return response()->json([
//             'height' => $jsonOutput['height'],
//             'weight' => $jsonOutput['weight'],
//             'bmi' => $bmi
//         ]);
        
//     }
    public function processImage(Request $request)
    {
 
        $validator = Validator::make($request->all(), [
            'image' => 'required|image|mimes:jpg,png,jpeg|max:2048',
        ]);
    
        // Agar validation fail hoti hai toh error response bhejein
        if ($validator->fails()) {
            return response()->json([
                'message' => $validator->errors()->first(),
                'errors' => $validator->errors()
            ], 422);
        }
        $image = $request->file('image');
        $imageName = time() . '.' . $image->extension();

        // 🔹 Image store karein
        $image->move(public_path('user/image'), $imageName);
        $imagePath = public_path('user/image/' . $imageName);

        // \Log::info("Stored Image Path: " . $imagePath);

        // ✅ Python Script Call
        // $pythonPath = "C:/Users/pc/myenv/Scripts/python.exe";
        // $scriptPath = "C:/Users/pc/Desktop/bmi-calculator/bmi-backend/scripts/extract_height_weight.py";
        // ✅ Dynamic Python Path from .env (fallback to 'python' if not set)
        $pythonPath = env('PYTHON_PATH', 'python'); 

        // ✅ Dynamic Script Path
        $scriptPath = base_path('scripts/extract_height_weight.py');
        
        $command = escapeshellcmd("\"$pythonPath\" \"$scriptPath\" \"$imagePath\"");
        $output = shell_exec($command);

        // \Log::info("Python script raw output: " . $output);

        // 🔹 Python output check karo
        $lines = explode("\n", trim($output));
        $jsonString = end($lines);
        $jsonOutput = json_decode($jsonString, true);

        if ($jsonOutput === null) {
            return response()->json([
                'error' => 'Invalid JSON response from Python script',
                'raw_output' => $output
            ], 500);
        }

        // ✅ Python Response Validation
        if (!isset($jsonOutput['height']) || !isset($jsonOutput['weight'])) {
            return response()->json([
                'error' => 'No human detected in the image. Please upload a clear image of a person.',
                'raw_output' => $jsonOutput
            ], 400);
        }


        // ✅ BMI Calculation
        $height = $jsonOutput['height'];
        $weight = $jsonOutput['weight'];
        $bmi = round($weight / pow(($height / 100), 2), 2);

        // ✅ AI API Call - Gemini
        if (!file_exists($imagePath)) {
            return response()->json(['error' => 'Image file not found'], 500);
        }

        $base64Image = base64_encode(file_get_contents($imagePath));
        $mimeType = mime_content_type($imagePath);

        $prompt = "Analyze the given image and based on height: {$height} cm, weight: {$weight} kg, and BMI: {$bmi}, provide a structured health recommendation.

        Strictly format the response as:

        Estimated BMI: **{$bmi}**

        Overview:
        (Short summary of key health points)

        Diet:
        - (Point 1)
        - (Point 2)
        - (Point 3)

        Workout:
        - (Point 1)
        - (Point 2)
        - (Point 3)

        Lifestyle:
        - (Point 1)
        - (Point 2)

        Important Considerations:
        - (Point 1)
        - (Point 2)

        Only use dashes (-) at the start of each point. Do NOT use asterisks (*).";

        $apiKey = config('app.GOOGLE_GEMINI_API_KEY');
        $url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=$apiKey";

        $response = Http::withHeaders(['Content-Type' => 'application/json'])->post($url, [
            "contents" => [
                [
                    "parts" => [
                        ["text" => $prompt],
                        ["inlineData" => [
                            "mimeType" => $mimeType,
                            "data" => $base64Image
                        ]]
                    ]
                ]
            ]
        ]);

        if ($response->failed()) {
            return response()->json(['error' => 'Google Gemini API request failed.', 'details' => $response->json()], 400);
        }

        $data = $response->json();

        if (!isset($data['candidates'][0]['content']['parts'][0]['text'])) {
            return response()->json(['error' => 'Invalid response from Google Gemini API.', 'details' => $data], 400);
        }

        $aiResponse = $data['candidates'][0]['content']['parts'][0]['text'];

        // ✅ Extract Estimated BMI using regex
        preg_match('/Estimated BMI:\s*\**([\d\.]+)\**/i', $aiResponse, $bmiMatch);
        $extractedBmi = isset($bmiMatch[1]) ? trim($bmiMatch[1]) : '';

        // ✅ Fallback to Python-detected BMI
        $estimatedBmi = ($extractedBmi !== '') ? $extractedBmi : $bmi;

        // ✅ AI Response Parsing
        $sections = preg_split('/\n*(Overview|Diet|Workout|Lifestyle|Important Considerations):\n*/', $aiResponse, -1, PREG_SPLIT_DELIM_CAPTURE);

        function formatBulletPoints($text)
        {
            $lines = explode("\n", $text);
            $formattedLines = array_map(fn($line) => trim($line, "- "), $lines);
            return array_values(array_filter($formattedLines, fn($line) => !empty($line)));
        }

        $structuredResponse = [
            'estimated_bmi' => $estimatedBmi,
            'overview' => trim($sections[2] ?? 'No overview available.'),
            'diet' => formatBulletPoints($sections[4] ?? ''),
            'workout' => formatBulletPoints($sections[6] ?? ''),
            'lifestyle' => formatBulletPoints($sections[8] ?? ''),
            'important_considerations' => formatBulletPoints($sections[10] ?? '')
        ];

        return response()->json([
            'height' => $height,
            'weight' => $weight,
            'bmi' => $bmi,
            'recommendation' => $structuredResponse
        ]);
    }


}
