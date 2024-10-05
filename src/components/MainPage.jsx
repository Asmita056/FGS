// main page
import React, { useState } from "react";
import DefaultImage from "../images/default_img.png";
import UploadImage from "../images/upload_img.png";

export default function MainPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [input, setInput] = useState("");
  const [predictions, setPredictions] = useState(null);
  const [accuracy_percent, setAccuracy] = useState(null);
  const [bestCount, setBestCount] = useState(0);
  const [averageCount, setAverageCount] = useState(0);
  const [worstCount, setWorstCount] = useState(0);
  const [totalCount, setTotalCount] = useState(0);

  const fileChangeHandler = (e) => {
    setSelectedFile(e.target.files[0]);
  };

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
    }
  };

  const handleSubmit = (e) => {
    e.preventDefault();

    if (!(selectedFile instanceof File)) {
      console.error("selectedFile is not a valid File object");
      return;
    }

    const formData = new FormData();
    formData.append("file", selectedFile);
    formData.append("username", input);

    const requestOptions = {
      method: "POST",
      body: formData,
    };

    fetch("http://127.0.0.1:8000/upload", requestOptions)
      .then((response) => response.json())
      .then((data) => {
        console.log("Response Data:", data);
        setPredictions(data.predictions);
        setTotalCount(data.total_count);
        setBestCount(data.best_count);
        setAverageCount(data.average_count);
        setWorstCount(data.worst_count);
      })
      .catch((error) => console.error("Error:", error));
  };

  return (
    <div className="flex flex-col items-center bg-[#F3FFCF] p-3">
      <div className="h-96 w-96">
        <img
          src={selectedFile ? URL.createObjectURL(selectedFile) : UploadImage}
          alt="Upload Preview"
          className="h-full w-full object-cover"
        />
      </div>

      <form
        id="form"
        encType="multipart/form-data"
        className="m-2"
        onSubmit={handleSubmit}
      >
        <input
          type="file"
          id="file-input"
          name="file"
          onChange={handleImageChange}
          className="m-3"
          accept=".jpeg, .png, .jpg"
        />
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder="Enter username"
          className="m-3 p-2 border rounded"
        />
        <button
          type="submit"
          className="bg-blue-500 p-3 rounded-lg hover:bg-blue-600"
        >
          Confirm
        </button>
      </form>
      {/* {predictions && (
        <div className="flex flex-col items-center mt-3">
          <div className="font-semibold text-lg">The current fruit is:</div>
          <div className="flex">
            <div className="shadow-lg m-4 p-3 flex flex-col rounded-xl">
              {Object.defineProperties(predictions).map(([model, result]) => (
                <span key={model}>
                  {model}: {result}
                </span>
              ))}
              <span>Category</span>
              <span className="text-center">Best</span>
            </div>
            <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
              <span>Accuracy</span>
              <span className="text-center">80</span>
            </div>
          </div>
        </div>
      )} */}
      {predictions && (
        <div className="flex mx-auto my-auto">
          <div className="flex flex-col items-center m-2">
            <div className="font-semibold text-lg">Predictions</div>
            <div className="shadow-lg m-4 p-3 flex flex-col rounded-xl">
              {/* {Object.entries(predictions).map(([model, result]) => (
              <span key={model}>
                {model}: {result}
              </span>
            ))} */}
              <span>Category: {predictions}</span>
            </div>
          </div>

          {/* <div className="flex flex-col items-center mt-2">
            <div className="font-semibold text-lg">Accuracy</div>
            <div className="shadow-lg m-4 p-3 flex flex-col rounded-xl">
              <span>Percent: {accuracy_percent}</span>
            </div>
          </div> */}
        </div>
      )}

      <div className="flex flex-col items-center mt-4">
        <p className="text-lg font-semibold">Today's Analysis</p>
        <div className="flex justify-center ">
          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>No. of Fruits</span>
            <span className="text-center">{totalCount}</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Best Quality</span>
            <span className="text-center">{bestCount}</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Average Quality</span>
            <span className="text-center">{averageCount}</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Worst Quality</span>
            <span className="text-center">{worstCount}</span>
          </div>
        </div>
      </div>
    </div>
  );
}