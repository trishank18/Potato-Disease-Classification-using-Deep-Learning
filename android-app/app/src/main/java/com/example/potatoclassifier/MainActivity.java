package com.example.potatoclassifier;

import android.app.Activity;
import android.content.Intent;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.os.Bundle;
import android.view.Gravity;
import android.view.ViewGroup;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.LinearLayout;
import android.widget.TextView;

import org.tensorflow.lite.Interpreter;

import java.io.FileInputStream;
import java.io.InputStream;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends Activity {
    private static final int PICK_IMAGE = 1;
    private static final int IMAGE_SIZE = 256;
    private static final String[] CLASSES = {
            "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"
    };

    private Interpreter interpreter;
    private ImageView imageView;
    private TextView resultView;

    @Override
    public void onCreate(Bundle state) {
        super.onCreate(state);
        buildScreen();
        try {
            interpreter = new Interpreter(loadModel());
        } catch (Exception e) {
            resultView.setText("Could not load the classifier model.");
        }
    }

    private void buildScreen() {
        LinearLayout layout = new LinearLayout(this);
        layout.setOrientation(LinearLayout.VERTICAL);
        layout.setGravity(Gravity.CENTER_HORIZONTAL);
        layout.setPadding(32, 48, 32, 32);

        TextView title = new TextView(this);
        title.setText("Potato Disease Classification");
        title.setTextSize(24);
        title.setGravity(Gravity.CENTER);
        layout.addView(title, new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT));

        TextView intro = new TextView(this);
        intro.setText("Choose a potato leaf photo to classify it as early blight, late blight, or healthy.");
        intro.setTextSize(16);
        intro.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams introParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        introParams.topMargin = 20;
        layout.addView(intro, introParams);

        Button chooseButton = new Button(this);
        chooseButton.setText("Choose image");
        chooseButton.setOnClickListener(view -> chooseImage());
        LinearLayout.LayoutParams buttonParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.WRAP_CONTENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        buttonParams.topMargin = 24;
        layout.addView(chooseButton, buttonParams);

        resultView = new TextView(this);
        resultView.setTextSize(18);
        resultView.setGravity(Gravity.CENTER);
        LinearLayout.LayoutParams resultParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, ViewGroup.LayoutParams.WRAP_CONTENT);
        resultParams.topMargin = 24;
        layout.addView(resultView, resultParams);

        imageView = new ImageView(this);
        imageView.setAdjustViewBounds(true);
        imageView.setScaleType(ImageView.ScaleType.FIT_CENTER);
        LinearLayout.LayoutParams imageParams = new LinearLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT, 0, 1f);
        imageParams.topMargin = 20;
        layout.addView(imageView, imageParams);

        setContentView(layout);
    }

    private void chooseImage() {
        Intent intent = new Intent(Intent.ACTION_GET_CONTENT);
        intent.setType("image/*");
        startActivityForResult(Intent.createChooser(intent, "Select a leaf image"), PICK_IMAGE);
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode != PICK_IMAGE || resultCode != RESULT_OK || data == null || data.getData() == null) return;
        try (InputStream stream = getContentResolver().openInputStream(data.getData())) {
            Bitmap bitmap = BitmapFactory.decodeStream(stream);
            if (bitmap == null) throw new IllegalArgumentException("Unsupported image");
            imageView.setImageBitmap(bitmap);
            if (interpreter == null) {
                resultView.setText("Classifier model is unavailable.");
                return;
            }
            classify(bitmap);
        } catch (Exception e) {
            resultView.setText("Could not read that image. Please choose another.");
        }
    }

    private void classify(Bitmap bitmap) {
        Bitmap resized = Bitmap.createScaledBitmap(bitmap, IMAGE_SIZE, IMAGE_SIZE, true);
        ByteBuffer input = ByteBuffer.allocateDirect(1 * IMAGE_SIZE * IMAGE_SIZE * 3 * 4);
        input.order(ByteOrder.nativeOrder());
        int[] pixels = new int[IMAGE_SIZE * IMAGE_SIZE];
        resized.getPixels(pixels, 0, IMAGE_SIZE, 0, 0, IMAGE_SIZE, IMAGE_SIZE);
        for (int pixel : pixels) {
            input.putFloat(((pixel >> 16) & 0xff));
            input.putFloat(((pixel >> 8) & 0xff));
            input.putFloat(pixel & 0xff);
        }
        float[][] output = new float[1][CLASSES.length];
        interpreter.run(input, output);

        int best = 0;
        for (int i = 1; i < output[0].length; i++) {
            if (output[0][i] > output[0][best]) best = i;
        }
        resultView.setText(String.format("Prediction: %s\nConfidence: %.2f%%", CLASSES[best], output[0][best] * 100));
    }

    private MappedByteBuffer loadModel() throws Exception {
        android.content.res.AssetFileDescriptor descriptor = getAssets().openFd("potato_model.tflite");
        try (FileInputStream input = new FileInputStream(descriptor.getFileDescriptor())) {
            FileChannel channel = input.getChannel();
            return channel.map(FileChannel.MapMode.READ_ONLY, descriptor.getStartOffset(), descriptor.getLength());
        } finally {
            descriptor.close();
        }
    }

    @Override
    protected void onDestroy() {
        if (interpreter != null) interpreter.close();
        super.onDestroy();
    }
}
