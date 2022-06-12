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
import android.util.Log
import android.view.View
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.content.FileProvider
import com.example.finalproject.databinding.ActivityMainBinding
//import com.example.finalproject.ml.MobilenetV110224Quant
import com.example.finalproject.ml.PoseClassifier
import org.tensorflow.lite.DataType
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import java.io.File
import java.nio.ByteBuffer
import java.nio.ByteOrder

//for getMax Replacement
import kotlin.math.*

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
                val imgHeight = 300
                val imgWidth = 300

                val resized = Bitmap.createScaledBitmap(bitmap, imgWidth, imgHeight, true)
                val model = PoseClassifier.newInstance(this)
                val tbuffer = TensorImage.fromBitmap(resized)
                val tensorImage = tbuffer.load(resized)
                println("Allocating ByteBuffer")
                //val byteBuffer = tbuffer.buffer
                /*

                0

first I need to ask you, did you use optimization when you convert the model to tfLite? if yes than you should read from here https://www.tensorflow.org/lite/performance/post_training_quantization. it is said that the optimize version is 4 times smaller so you need to create your byte buffer to [4 * 224 * 224 * 3] so it can match with the inputfeature0 and i dont know what kind of your input is it image or ju
                 */
                val byteBuffer = ByteBuffer.allocateDirect(4 * imgHeight * imgWidth * 3 )//we have to manually allocate this because it is not quantized
// Creates inputs for reference.
                println("Initiating Feature Input with Float32")
                // print a logcat
                val inputFeature0 =
                    TensorBuffer.createFixedSize(intArrayOf(1, imgWidth, imgHeight, 3), DataType.FLOAT32)

                //val  inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(180, 180, 3), DataType.FLOAT32)
                //val inputFeature0 = TensorBuffer.createFixedSize(intArrayOf(1, 244, 244, 3), DataType.UINT8) // we're back into the default mobilenet thingy hahahaahhahah
                println("Loading ByteBuffer")

                byteBuffer.order(ByteOrder.nativeOrder())
                //https://www.youtube.com/watch?v=jhGm4KDafKU 9:13
                println(byteBuffer)
                inputFeature0.loadBuffer(byteBuffer)
                println("Loaded Bytebuffer")

// Runs model inference and gets result.
                val outputs = model.process(inputFeature0)
                println("Output processed")
                val outputFeature0 = outputs.outputFeature0AsTensorBuffer
                println("getting Max")
                val max = getMax(outputFeature0.floatArray)
                println("ok")
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
        /*
        var ind = 0;
        var min = 0.0f;
        // This is for searching the highest value out of all index
        //for(i in 0..1000)
        for(i in 0..5)
        {
                println("index getMax debug: ")
                print(i)
                println("modelPredOutput: ")
                print(arr[i])
                println()
                if (arr[i] >= min) {
                    println("higher than min")
                    print(i)
                    print(arr[i])
                    min = arr[i]
                    ind = i
            }
        }
        man this code is uhh...
         */
        var ind = 0;
        var min = 0.0f;
        val maxArrayAccess = arr.count() - 1 //prevent seeking out of the bounds of the array size number from 0 to 4 or in total 5 elements 0 1 2 3 4
        println("array size")
        print(maxArrayAccess)
        for(i in 0..maxArrayAccess)
        {
            println("checkingArray")
            print(i)
            if( arr[i] > min ){
                if( i < 1 ) {
                    print("noComparison")
                    ind=i
                }else {
                if( arr[i] > arr[i-1] && arr[i] > arr[ind]){
                    ind=i
                    print(arr[i])
                    println("index Max found")
                    print(i)
                }
                }
            }
        }
        println("result Max")
        print(ind)
        return ind
    }

}