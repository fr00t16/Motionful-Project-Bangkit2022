package com.example.finalproject

import android.Manifest
import android.app.Activity
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import androidx.appcompat.app.AppCompatActivity
import android.os.Bundle
import android.provider.MediaStore
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.finalproject.databinding.ActivityMainBinding
import com.example.finalproject.ml.MobilenetV110224Quant
import com.example.finalproject.ml.PoseClassifier
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File

class MainActivity : AppCompatActivity() {
    private lateinit var binding: ActivityMainBinding
    private lateinit var currentPhotoPath: String

    private var getFile: File? = null
    private lateinit var bitmap: Bitmap
    private var bitmatcod : Boolean = false
    private lateinit var uri: Uri

    lateinit var text_view : TextView
    lateinit var img_view : ImageView


    companion object {
        const val CAMERA_X_RESULT = 200

        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
        private const val REQUEST_CODE_PERMISSIONS = 10

    }

    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == REQUEST_CODE_PERMISSIONS) {
            if (!allPermissionsGranted()) {
                Toast.makeText(
                    this,
                    "Please go into the permission control panel",
                    Toast.LENGTH_SHORT
                ).show()
                finish()
            }
        }
    }

    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(baseContext, it) == PackageManager.PERMISSION_GRANTED
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)

        if (!allPermissionsGranted()) {
            ActivityCompat.requestPermissions(
                this,
                REQUIRED_PERMISSIONS,
                REQUEST_CODE_PERMISSIONS
            )
        }

        text_view = findViewById(R.id.yogaText)
        img_view = findViewById(R.id.previewImageView)


        val labels = application.assets.open("labels.txt").bufferedReader().use { it.readText() }.split("\n")

        binding.cameraButton.setOnClickListener(View.OnClickListener {
            var camera : Intent = Intent(android.provider.MediaStore.ACTION_IMAGE_CAPTURE)
            startActivityForResult(camera, 200)
        })

        binding.galleryButton.setOnClickListener {
            val intent = Intent()
            intent.action = Intent.ACTION_GET_CONTENT
            intent.type = "image/*"

            startActivityForResult(intent, 100)
        }
        binding.uploadButton.setOnClickListener {
            if (bitmatcod) {
                val resized = Bitmap.createScaledBitmap(bitmap, 180, 180, true)
                val model = PoseClassifier.newInstance(this)

                /*
                E/AndroidRuntime: FATAL EXCEPTION: main
    Process: com.example.finalproject, PID: 9626
    java.lang.IllegalArgumentException: The size of byte buffer and the shape do not match.
        at org.tensorflow.lite.support.common.SupportPreconditions.checkArgument(SupportPreconditions.java:104)
        at org.tensorflow.lite.support.tensorbuffer.TensorBuffer.loadBuffer(TensorBuffer.java:309)
        at org.tensorflow.lite.support.tensorbuffer.TensorBuffer.loadBuffer(TensorBuffer.java:328)
        at com.example.finalproject.MainActivity.onCreate$lambda-4(MainActivity.kt:115)
        at com.example.finalproject.MainActivity.$r8$lambda$3Q2p7gznKNYR-PF2arACFh7N1J8(Unknown Source:0)
        at com.example.finalproject.MainActivity$$ExternalSyntheticLambda2.onClick(Unknown Source:4)
        at android.view.View.performClick(View.java:7441)
        at com.google.android.material.button.MaterialButton.performClick(MaterialButton.java:1194)
        at android.view.View.performClickInternal(View.java:7418)
        at android.view.View.access$3700(View.java:835)
        at android.view.View$PerformClick.run(View.java:28676)
        at android.os.Handler.handleCallback(Handler.java:938)
        at android.os.Handler.dispatchMessage(Handler.java:99)
        at android.os.Looper.loopOnce(Looper.java:201)
        at android.os.Looper.loop(Looper.java:288)
        at android.app.ActivityThread.main(ActivityThread.java:7839)
        at java.lang.reflect.Method.invoke(Native Method)
        at com.android.internal.os.RuntimeInit$MethodAndArgsCaller.run(RuntimeInit.java:548)
        at com.android.internal.os.ZygoteInit.main(ZygoteInit.java:1003)
                 */
                val tbuffer = TensorImage.fromBitmap(resized)
                val byteBuffer = tbuffer.buffer
// Creates inputs for reference.
                /*
                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, 224, 224, 3), DataType.UINT8)
                */
                //val  inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(180, 180, 3), DataType.FLOAT32)
                val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 244, 244, 3), DataType.UINT8) // we're back into the default mobilenet thingy hahahaahhahah
                inputFeature0.loadBuffer(byteBuffer)

// Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer

                val max = getMax(outputFeature0.floatArray)

                text_view.setText(labels[max])

                model.close()

            }

            else if (!bitmatcod) Toast.makeText(this, "Please Insert the Image", Toast.LENGTH_SHORT).show()

        }
    }

    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)

        if (data!=null) {
            binding.previewImageView.setImageURI(data.data)

            uri = data.data!!
            bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            bitmatcod = true
        }
        else if(requestCode == 200 && resultCode == Activity.RESULT_OK){
            bitmap = data?.extras?.get("data") as Bitmap
            img_view.setImageBitmap(bitmap)
        }

    }

    fun getMax(arr:FloatArray) : Int{
        var ind = 0;
        var min = 0.0f;

        for(i in 0..1000)
        {
            if(arr[i] > min)
            {
                min = arr[i]
                ind = i;
            }
        }
        return ind
    }

}