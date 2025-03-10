<?php

namespace App\Http\Controllers;

use App\Http\Controllers\Controller;
use Illuminate\Http\Request;
use Illuminate\Support\Facades\Http;
use Illuminate\Support\Facades\Storage;
use Symfony\Component\Process\Process;
use Symfony\Component\Process\Exception\ProcessFailedException;
use Illuminate\Support\Facades\Log;
class AIController extends Controller
{
    public function getHealthRecommendation(Request $request)
    {
        $bmi = $request->input('bmi');
        $age = $request->input('age');
        $gender = $request->input('gender');
    
        // ✅ Refined AI Prompt
        $prompt = "Provide a health recommendation for a $age-year-old $gender with BMI $bmi.
                   Format the response as:
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
    
                   Do NOT use asterisks (*) for bullet points. Only use dashes (-) at the start of each point.";
    
        $apiKey = config('app.GOOGLE_GEMINI_API_KEY');
        if (!$apiKey) {
            return response()->json(['error' => 'Google Gemini API Key is missing.'], 400);
        }
    
        $url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=$apiKey";
    
        $response = Http::withHeaders(['Content-Type' => 'application/json'])->post($url, [
            "contents" => [["parts" => [["text" => $prompt]]]]
        ]);
    
        if ($response->failed()) {
            return response()->json(['error' => $response->json(), 'message' => 'Google Gemini API request failed.'], 400);
        }
    
        $data = $response->json();
        \Log::info('Gemini API Response:', $data); // Debug API response
    
        if (!isset($data['candidates'][0]['content']['parts'][0]['text'])) {
            return response()->json(['error' => $data, 'message' => 'Invalid response from Google Gemini API.'], 400);
        }
    
        $aiResponse = $data['candidates'][0]['content']['parts'][0]['text'];
        \Log::info('Raw AI Response:', ['text' => $aiResponse]); // Debug AI text
    
        // ✅ Improved Regex to Match Different AI Formats
        $sections = preg_split('/\n*(Overview|Diet|Workout|Lifestyle|Important Considerations):\n*/', $aiResponse, -1, PREG_SPLIT_DELIM_CAPTURE);
    
        function formatBulletPoints($text)
        {
            $lines = explode("\n", $text);
            $formattedLines = array_map(fn($line) => trim($line, "- "), $lines); // Remove leading `-` from each point
            return array_values(array_filter($formattedLines, fn($line) => !empty($line))); // Remove empty lines
        }
    
        $structuredResponse = [
            'overview' => trim($sections[2] ?? 'No overview available.'),
            'diet' => formatBulletPoints($sections[4] ?? ''),
            'workout' => formatBulletPoints($sections[6] ?? ''),
            'lifestyle' => formatBulletPoints($sections[8] ?? ''),
            'important_considerations' => formatBulletPoints($sections[10] ?? '')
        ];
    
        return response()->json(['recommendation' => $structuredResponse]);
    }

    // CHATBOT AI FUNCTION HERE
    public function chatWithAI(Request $request)
    {
        // return env('GOOGLE_GEMINI_API_KEY');
        // return config('app.GOOGLE_GEMINI_API_KEY');
        $userMessage = $request->input('message');

        if (!$userMessage) {
            return response()->json(['error' => 'User message is required.'], 400);
        }

        $apiKey = config('app.GOOGLE_GEMINI_API_KEY');
        if (!$apiKey) {
            return response()->json(['error' => 'Google Gemini API Key is missing.'], 400);
        }

        // ✅ Improved Prompt for Short and Direct Answers
        // $prompt = "You are a health AI chatbot. 
        // - Keep your responses short (2-3 sentences).
        // - If asked about BMI, diet, or workouts, provide structured advice in simple words.
        // - Avoid long paragraphs, just give useful and to-the-point advice.
        // - If the question is unclear, ask for clarification.
        
        // User: $userMessage
        $prompt = "You are a knowledgeable, friendly, and concise health AI chatbot.
        - Respond in 2-3 short sentences.
        - Provide structured, actionable advice on topics like BMI, diet, and workouts, using bullet points if needed.
        - Avoid long paragraphs and unnecessary details.
        - If the user's question is ambiguous, ask for clarification—but NEVER ask for details (such as weight, height, or activity level) if they have already been provided in previous messages.
        - When the user has provided such details, calculate the BMI using the formula (BMI = weight (kg) / (height (m))^2) and use that information to give a tailored, personalized response.
        - For vegetarian meal queries, do not include eggs unless the term 'eggetarian' is explicitly mentioned.
        - Always maintain a supportive, professional tone.
        
        User: $userMessage
        AI:";
        $url = "https://generativelanguage.googleapis.com/v1/models/gemini-1.5-flash:generateContent?key=$apiKey";

        $response = Http::withHeaders(['Content-Type' => 'application/json'])->post($url, [
            "contents" => [["parts" => [["text" => $prompt]]]],
            "generationConfig" => [
                "maxOutputTokens" => 150 // ✅ Shortens AI response
            ]
        ]);

        if ($response->failed()) {
            return response()->json(['error' => 'Google Gemini API request failed.'], 400);
        }

        $data = $response->json();
        
        if (!isset($data['candidates'][0]['content']['parts'][0]['text'])) {
            return response()->json(['error' => 'Invalid response from Google Gemini API.'], 400);
        }

        // ✅ Trimmed and Cleaned AI Response
        $aiResponse = trim($data['candidates'][0]['content']['parts'][0]['text']);

        return response()->json(['response' => $aiResponse]);
    }
    


    public function getImageHealthRecommendation(Request $request)
    {
        if (!$request->hasFile('image')) {
            return response()->json(['error' => 'Image is required.'], 400);
        }

        $imageFile = $request->file('image');
        $mimeType = $imageFile->getMimeType();

        if (!in_array($mimeType, ['image/jpeg', 'image/png', 'image/webp'])) {
            return response()->json(['error' => 'Invalid image format. Only JPEG, PNG, and WEBP are allowed.'], 400);
        }

        $imagePath = $request->file('image')->store('bmi_images', 'public');
        $imageFullPath = storage_path('app/public/' . $imagePath);

        $process = new Process(['python', base_path('bmi_ai/bmi_estimator.py'), $imageFullPath]);
        $process->run();

        if (!$process->isSuccessful()) {
            return response()->json([
                'error' => 'BMI Calculation Failed',
                'python_error' => trim($process->getErrorOutput())
            ], 500);
        }

        $bmiOutput = json_decode(trim($process->getOutput()), true);

        if (isset($bmiOutput['error'])) {
            return response()->json([
                'error' => 'No human detected. Please upload a clear, full-body image.',
                'details' => $bmiOutput
            ], 400);
        }

        // Extract height, weight, BMI from Python response
        $height = $bmiOutput['height_cm'];
        $weight = $bmiOutput['weight_kg'];
        $bmi = $bmiOutput['bmi'];

        // Google Gemini API Integration
        $base64Image = base64_encode(file_get_contents($imageFile->path()));

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

        $apiKey = config('app.GOOGLE_GEMINI_KEY');
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
            return response()->json(['error' => $response->json(), 'message' => 'Google Gemini API request failed.'], 400);
        }

        $data = $response->json();

        if (!isset($data['candidates'][0]['content']['parts'][0]['text'])) {
            return response()->json(['error' => $data, 'message' => 'Invalid response from Google Gemini API.'], 400);
        }

        $aiResponse = $data['candidates'][0]['content']['parts'][0]['text'];

        // Extract Estimated BMI using improved regex
        preg_match('/Estimated BMI:\s*\**([\d\.]+)\**/i', $aiResponse, $bmiMatch);
        $extractedBmi = isset($bmiMatch[1]) ? trim($bmiMatch[1]) : '';

        // If AI fails to detect BMI, fallback to Python-detected BMI
        $estimatedBmi = ($extractedBmi !== '') ? $extractedBmi : $bmi;

        // Parsing AI response into structured format
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

        return response()->json(['recommendation' => $structuredResponse]);
    }

    
}