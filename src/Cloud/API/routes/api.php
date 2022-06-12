<?php

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Route;

/**
 * route "/register"
 * @method "POST"
 */
Route::post('/register', App\Http\Controllers\Api\RegisterController::class)->name('register');
/**
 * route "/login"
 * @method "POST"
 */
Route::post('/login', App\Http\Controllers\Api\LoginController::class)->name('login');

/**
 * route "/user"
 * @method "GET"
 */
Route::middleware('auth:api')->get('/user', function (Request $request) {
    return $request->user();
});