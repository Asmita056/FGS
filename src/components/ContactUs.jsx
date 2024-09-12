import BgImage from "../images/contact-bg.jpeg";

export default function ContactUs() {
  return (
    <>
      <div className="bg-[#F3FFCF] min-h-screen">
        <div
          className="bg-cover bg-center h-80"
          style={{ backgroundImage: `url(${BgImage})` }}
        >
          {/* <h2 className="text-center font-bold text-4xl mb-4">Contact Us</h2> */}
        </div>
        <div className="p-3">
          {/* <div className="flex  justify-between ">
          <div className="card ">
            <h2>Operating Hours</h2>
            <h4>
              Morning, 9:00 A.M
              <br />
              to Evening, 5:00 P.M
            </h4>
          </div>
          <div className="card">
            <h2>Phone</h2>
            <h4>
              +91 9875579095 <br />
              +91 8907656783
            </h4>
          </div>
          <div className="card">
            <h2>General Enquiries</h2>
            <h4>phalSense.AI@gmail.com</h4>
          </div>
        </div> */}
          <div className="flex justify-center space-x-8 py-8">
            <div className="card  p-6 shadow-md rounded-lg w-64">
              <h2 className="text-[#143601] text-xl font-semibold">
                Operating Hours
              </h2>
              <h4 className="text-[#2E8D49]">
                Morning, 9:00 A.M
                <br />
                to Evening, 5:00 P.M
              </h4>
            </div>
            <div className="card  p-6 shadow-md rounded-lg w-64">
              <h2 className="text-[#143601] text-xl font-semibold">Phone</h2>
              <h4 className="text-[#2E8D49]">
                +91 9875579095 <br />
                +91 8907656783
              </h4>
            </div>
            <div className="card  p-6 shadow-md rounded-lg w-64">
              <h2 className="text-[#143601] text-xl font-semibold">
                General Enquiries
              </h2>
              <h4 className="text-[#2E8D49]">phalSense.AI@gmail.com</h4>
            </div>
          </div>
          {/* <hr className='text-#59EAA8'/> */}
          <hr className="w-[63.063rem] h-[0.09rem] bg-[#59EAA8] border-0 mx-auto my-4" />

          <div className="text-center items-center text-lg">
            We'd love to hear from you!
            <br /> Whether you have a question, want to learn more about our
            Fruit Grading System,
            <br />
            or explore potential collaborations, feel free to reach out to us.
          </div>
        </div>

        {/* <div>Email: [your.email@example.com]</div>
      <div> Phone: [Your Phone Number] </div>
      <div>Address: [Your Address] </div>
      <div>
        Social Media: [Links to your social media profiles]
        <p>
          Connect with us and let’s work together to transform the future of
          fruit grading!
        </p>
      </div> */}
      </div>
    </>
  );
}
