package com.dicoding.picodiploma.mybottomnavigation.ui.dashboard

import android.content.Intent
import android.content.pm.PackageManager
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.widget.Button
import android.widget.VideoView
import androidx.core.app.ActivityCompat
import com.dicoding.picodiploma.mybottomnavigation.R
import java.util.jar.Manifest

class AddActivity : AppCompatActivity() {
    private val button: Button = findViewById(R.id.btn_capture)
    private val videoView : VideoView = findViewById(R.id.videoView)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_add)

        button.isEnabled = false

        if (ActivityCompat.checkSelfPermission(this, android.Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED)
            ActivityCompat.requestPermissions(this, arrayOf(android.Manifest.permission.CAMERA),1111)
        else
            button.isEnabled =true
        button.setOnClickListener{
            var i = Intent(MediaStore.ACTION_VIDEO_CAPTURE)
            startActivityForResult(i, 1111)
        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (requestCode == 1111){
            videoView.setVideoURI(data?.data)
            videoView.start()
        }
    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<out String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == 1111 && grantResults[0] == PackageManager.PERMISSION_GRANTED)
        {
            button.isEnabled = true
        }
    }
}