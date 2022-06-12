//package com.dicoding.picodiploma.mybottomnavigation.ui.dashboard
//
//import android.os.Bundle
//import android.view.LayoutInflater
//import android.view.View
//import android.view.ViewGroup
//import android.widget.Button
//import android.widget.TextView
//import androidx.core.app.ActivityCompat
//import androidx.fragment.app.Fragment
//import androidx.lifecycle.ViewModelProvider
//import androidx.viewbinding.BuildConfig
//import com.bumptech.glide.Glide
//import com.dicoding.picodiploma.mybottomnavigation.R
//import com.dicoding.picodiploma.mybottomnavigation.databinding.FragmentDashboardBinding
//
//
//class DashboardFragment : Fragment(R.layout.fragment_dashboard) {
//    private lateinit var dashboardViewModel: DashboardViewModel
//    private lateinit var binding: FragmentDashboardBinding
//
//    override fun onCreateView(
//        inflater: LayoutInflater,
//        container: ViewGroup?,
//        savedInstanceState: Bundle?
//    ): View? {
//        dashboardViewModel =
//            ViewModelProvider(this).get(DashboardViewModel::class.java)
//        val root = inflater.inflate(R.layout.fragment_home, container, false)
//        val button: Button = root.findViewById(R.id.btn_capture)
//
//        return root
//    }
//
//    override fun onViewCreated(view: View, savedInstanceState: Bundle?) {
//        super.onViewCreated(view, savedInstanceState)
//
//        setupPermission()
//        setupAction()
//
//
//    }
//
//    private fun setupPermission() {
//        if (!allPermissionsGrant()) {
//            ActivityCompat.requestPermissions(requireActivity(), PERMISSION_REQUIRED, REQUEST_CODE)
//        }
//
//    }
//
//    private fun setupAction() {
//        if (imgFile == null) {
//            Glide.with(this).load(imgFile).placeholder(R.drawable.ic_baseline_image_24)
//                .fallback(R.drawable.ic_baseline_image_24).into(binding.imgUpload)
//        }
//
//        binding.apply {
//            btnCapture.setOnClickListener {
//                accessCamera()
//            }
//
//        }
//    }
//
//
//}