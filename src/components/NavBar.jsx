import { Link } from "react-router-dom";
import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Logo from "../images/guava.jpeg";

export default function NavBar({ email, setIsAuthenticated }) {
  const [isDropdownOpen, setIsDropdownOpen] = useState(false);
  const navigate = useNavigate();

  const handleLogout = () => {
    // setIsAuthenticated(false);
    navigate("/", { replace: true });
  };

  return (
    <>
      <div className="bg-[#143601] text-[#B0F689] flex items-center justify-between px-12 py-3 w-full shadow-lg">
        <div className="flex items-center space-x-4">
          <img
            src={Logo}
            className=" w-10 h-10 flex items-center rounded-lg "
          />
          <div className=" text-3xl py-2 px-4">PhalSense AI</div>
        </div>
        <div>
          <ul className="flex justify-end space-x-8 text-lg">
            <li className="p-3 hover:underline">
              <Link to="/Home">HOME</Link>
            </li>
            {/* <li className="p-3 hover:underline">
              <Link to="/Analysis">ANALYSIS</Link>
            </li> */}
            <li className="p-3 hover:underline">
              <Link to="/AboutUs">ABOUT US</Link>
            </li>
            <li className="p-3 hover:underline">
              <Link to="/ContactUs">CONTACT US</Link>
            </li>
            <li
              className="relative p-3"
              onMouseEnter={() => setIsDropdownOpen(true)}
              onMouseLeave={() => setIsDropdownOpen(false)}
            >
              <span className="cursor-pointer">PROFILE</span>
              {isDropdownOpen && (
                <div className="absolute right-0 mt-2 w-48 bg-white text-black shadow-lg rounded-md">
                  <div className="p-2 border-b border-gray-300 text-gray-400">
                    <span>{email}</span>
                  </div>
                  <div
                    className="p-2 hover:bg-gray-200 cursor-pointer"
                    onClick={handleLogout}
                  >
                    Logout
                  </div>
                </div>
              )}
            </li>
          </ul>
        </div>
      </div>
    </>
  );
}
