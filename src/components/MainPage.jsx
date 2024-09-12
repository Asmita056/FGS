import React, { useState } from "react";
import DefaultImage from "../images/default_img.png";

export default function MainPage() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [input, setInput] = useState("");

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
      .then((data) => console.log(data))
      .catch((error) => console.error("Error:", error));
  };

  return (
    <div className="flex flex-col items-center bg-[#F3FFCF] p-3">
      <div className="h-96 w-96">
        <img
          src={selectedFile ? URL.createObjectURL(selectedFile) : DefaultImage}
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

      <div className="flex flex-col items-center mt-3">
        <div className="font-semibold text-lg">The current fruit is:</div>
        <div className="flex">
          <div className="shadow-lg m-4 p-3 flex flex-col rounded-xl">
            <span>Category</span>
            <span className="text-center">Best</span>
          </div>
          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Accuracy</span>
            <span className="text-center">80</span>
          </div>
        </div>
      </div>

      <div className="flex flex-col items-center mt-4">
        <p className="text-lg font-semibold">Today's Analysis</p>
        <div className="flex justify-center ">
          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>No. of Fruits</span>
            <span className="text-center">100</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Best Quality</span>
            <span className="text-center">80</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Average Quality</span>
            <span className="text-center">15</span>
          </div>

          <div className="shadow-lg m-4 p-3 flex-col flex rounded-xl">
            <span>Worst Quality</span>
            <span className="text-center">5</span>
          </div>
        </div>
      </div>
    </div>
  );
}
