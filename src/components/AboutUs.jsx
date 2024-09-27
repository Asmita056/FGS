import Boy from "../images/boy.jpeg";
import Girl from "../images/girl.jpg";

export default function AboutUs() {
  return (
    <>
      <div className="bg-[#F3FFCF] min-h-screen p-4">
        <h2 className="text-center font-bold text-4xl m-3">OUR PRODUCT</h2>
        <div className=" text-center text-[#2E8D49] text-lg">
          Welcome to PhalSense AI, where we innovate to elevate the fruit
          grading industry. Our mission is to harness cutting-edge technology to
          ensure only the highest quality produce reaches consumers while
          promoting sustainability. At PhalSense AI, we've developed the Fruit
          Grading System, a hybrid model that categorizes fruits into Best,
          Average, or Worst with exceptional accuracy. By integrating machine
          learning with traditional automation, our system enhances grading
          precision and efficiency beyond traditional methods.
          <br />
          Our setup involves a conveyor belt with high-resolution cameras
          capturing fruit images, which are then analysed by our advanced model.
          A real-time admin dashboard offers valuable insights for better
          decision-making and optimized resource use. We're committed to
          reducing waste and boosting productivity in the supply chain. Join us
          in shaping the future of agriculture, one fruit at a time.
        </div>
        <div className="mb-6 p-4">
          <div>
            <h3 className="text-center font-bold text-2xl m-4">OUR TEAM</h3>
          </div>
          <div className="flex justify-between px-20">
            <div className="m-5">
              <img src={Girl} alt="Purva" className="w-40 h-60" />
              <div className=" text-center text-xl text-[#143601] mt-5">
                Purva Nagap
              </div>
            </div>
            <div className="m-3">
              <img src={Boy} alt="Aditya" className="w-40 h-60" />
              <div className=" text-center text-xl text-[#143601] mt-5">
                Aditya Thorat
              </div>
            </div>
            <div className="m-3">
              <img src={Girl} alt="Eshwari" className="w-40 h-60" />
              <div className=" text-center text-xl text-[#143601] mt-5">
                Eshwari Rampoore
              </div>
            </div>
            <div className="m-3">
              <img src={Girl} alt="Asmita" className="w-40 h-60" />
              <div className=" text-center text-xl text-[#143601] mt-5">
                Asmita Ghode
              </div>
            </div>
          </div>
        </div>
      </div>
    </>
  );
}
