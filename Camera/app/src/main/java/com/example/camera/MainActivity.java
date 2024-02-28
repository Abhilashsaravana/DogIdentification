package com.example.camera;

import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import android.Manifest;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.os.Bundle;
import android.provider.MediaStore;
import android.view.View;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

public class MainActivity extends AppCompatActivity {
    ImageView selectedImage;
    Button Camera, Gallery; //Declares the camera and gallery button


    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        selectedImage = findViewById(R.id.imageView2);
        Camera = findViewById(R.id.Camera);
        Gallery = findViewById(R.id.Gallery);

        /*Asks for permission and displays a text when the camera button is clicked

         */
        Camera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Permissions();
                Toast.makeText(MainActivity.this, "Camera Button is Clicked.", Toast.LENGTH_SHORT).show();

            }
        });

        /* Displays a message when the gallery button is clicked

         */
        Gallery.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                Toast.makeText(MainActivity.this, "Gallery Button is Clicked.", Toast.LENGTH_SHORT).show();

            }
        });
    }
    /* Asks the user permission to open the camera

     */
    private void Permissions() {
        if (ContextCompat.checkSelfPermission(this, Manifest.permission.CAMERA) != PackageManager.PERMISSION_GRANTED) {
            ActivityCompat.requestPermissions(this, new String[]{android.Manifest.permission.CAMERA}, 101);
        } else {
            openCamera();
        }
    }

    /*Checks if permission to open the camera is already granted or not. If not, request is shown otherwise
    It calls the openCamera() function

     */
    //@Override
    public void onRequestPersmissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        if (requestCode == 101) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera();
            } else {
                Toast.makeText(this, "Request to Open Camera", Toast.LENGTH_SHORT).show();
            }
        }
    }

    //Opens the camera

    private void openCamera() {

        Intent camera = new Intent(MediaStore.ACTION_IMAGE_CAPTURE);
         startActivityForResult(camera, 102);
    }
}

