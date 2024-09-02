import DefaultImage from "../images/default_img.png";
import { useState } from "react";

export default function MainPage() {
  const [imageURL, setImageURL] = useState(DefaultImage);

  const handleImageChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        setImageURL(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  return (
    <div className="flex flex-col items-center">
      <div className="h-96 w-96">
        <img
          src={imageURL}
          alt="Upload Image"
          className="h-full w-full object-cover"
        />
      </div>

      <form id="form" encType="multipart/form-data" className="m-2">
        <input
          type="file"
          id="file-input"
          name="ImageStyle"
          onChange={handleImageChange}
          className="m-3"
        />
      </form>

      <div className="flex flex-col items-center">
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

        <div className="flex justify-center mt-4">
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
