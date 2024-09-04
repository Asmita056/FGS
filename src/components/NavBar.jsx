import { Link } from "react-router-dom";

export default function NavBar() {
  return (
    <>
      <div className="bg-[#143601] text-[#B0F689] flex items-center justify-between px-12 py-3 w-full shadow-lg">
        <div className="flex items-center space-x-4">
          <div className="bg-white w-10 h-10 flex items-center rounded-lg py-2 px-4"></div>
          <div className=" text-3xl py-2 px-4">
            PhalSense AI
          </div>
        </div>
        <div>
          <ul className="flex justify-end space-x-8 text-lg">
            <li className="p-3">
              <Link to="/">HOME</Link>
            </li>
            <li className="p-3">
              <Link to="/Analysis">ANALYSIS</Link>
            </li>
            <li className="p-3">
              <Link to="/AboutUs">ABOUT US</Link>
            </li>
            <li className="p-3">
              <Link to="/ContactUs">CONTACT US</Link>
            </li>
            <li className="p-3">
              <Link to="/Profile">PROFILE</Link>
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}
