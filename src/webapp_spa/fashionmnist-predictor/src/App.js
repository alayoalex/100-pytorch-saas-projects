// App.js
import React, { useState } from "react";
import axios from "axios";

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [prediction, setPrediction] = useState("");
  const [imagePreview, setImagePreview] = useState(null);

  // Handle file selection
  const onFileChange = (event) => {
    setSelectedFile(event.target.files[0]);
    setImagePreview(URL.createObjectURL(event.target.files[0]));
  };

  // Handle form submission (upload)
  const onFileUpload = async () => {
    if (!selectedFile) {
      alert("Please select an image first!");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: {
          "Content-Type": "multipart/form-data",
        },
      });
      console.log("Prediction result:", response.data);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error during prediction request:", error);
    }
  };

  return (
    <div className="App">
      <h1>FashionMNIST Prediction</h1>

      <input type="file" accept="image/png, image/jpeg" onChange={onFileChange} />

      {imagePreview && (
        <div>
          <h3>Uploaded Image:</h3>
          <img src={imagePreview} alt="Preview" width="200px" />
        </div>
      )}

      <button onClick={onFileUpload}>Upload and Predict</button>

      {prediction && (
        <div>
          <h2>Prediction Result: {prediction.class_name}</h2>
        </div>
      )}
    </div>
  );
}

export default App;
