<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;
use App\Http\Controllers\AIController;
use App\Http\Controllers\ImageController;
Route::get('/user', function (Request $request) {
    return $request->user();
})->middleware('auth:sanctum');

Route::post('/get-ai-recommendation', [AIController::class, 'getRecommendation']);
Route::post('/chat-with-ai',[AIController::class,'chatWithAI']);
Route::post('/bmi/recommendation',[AIController::class,'getHealthRecommendation']);
Route::post('/health-image-recommendation', [ImageController::class, 'processImage']);