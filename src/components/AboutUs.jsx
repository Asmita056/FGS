import Boy from "../images/boy.jpeg";
import Girl from "../images/girl.jpg";

export default function AboutUs() {
  return (
    <>
      <h2 className="text-center font-bold text-4xl m-4">About Us</h2>
      <div className=" p-4">
        At CodeQuest, we are dedicated to revolutionizing the fruit industry
        through innovative technology solutions. Our state-of-the-art fruit
        grading system leverages advanced machine learning algorithms and
        computer vision to ensure the highest standards of quality control. By
        automating the grading process, we help producers and distributors
        achieve consistent, efficient, and accurate results, reducing waste and
        enhancing profitability. With a strong commitment to sustainability and
        technological excellence, we strive to set new benchmarks in the
        industry, empowering businesses to deliver the best produce to the
        market.
      </div>
      <div className="m-4 p-4">
        <div className="font-semibold text-xl">Our Team</div>
        <div className="flex justify-between">
          <div className="m-5">
            <div className="mb-5">Purva Nagap</div>
            <img src={Girl} alt="Purva" className="w-40 h-60" />
          </div>
          <div className="m-3">
            <div className="mb-5">Aditya Thorat</div>
            <img src={Boy} alt="Aditya" className="w-40 h-60" />
          </div>
          <div className="m-3">
            <div className="mb-5">Eshwari Rampoore</div>
            <img src={Girl} alt="Eshwari" className="w-40 h-60" />
          </div>
          <div className="m-3">
            <div className="mb-5">Asmita Ghode</div>
            <img src={Girl} alt="Asmita" className="w-40 h-60" />
          </div>
        </div>
      </div>
    </>
  );
}
