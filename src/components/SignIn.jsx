import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import Logo from "../images/guava.jpeg";

export default function SignIn() {
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const navigate = useNavigate();

  const handleSubmit = async (e) => {
    e.preventDefault();

    if (email == "123" && password == "123") {
      // window.history.replaceState(null, null, "/Home");
      // window.location.replace = "http://localhost:5173/Home";
      navigate("/Home", { replace: true });
    }
  };

  return (
    <div className="h-screen flex justify-center items-center mx-auto my-auto bg-[#F3FFCF]">
      <div className="flex items-center h-4/6 justify-center rounded-3xl bg-[#f2f7e6] w-3/4 shadow-lg">
        <div className="flex flex-col justify-center w-1/2 items-center m-4">
          <img src={Logo} className="w-32 rounded-full" />
          <div className="text-7xl">PhalSense AI</div>
        </div>

        <div className="mx-8 h-5/6 w-0.5 bg-gray-300"></div>

        {/* second half */}
        <div className="w-1/2 flex flex-col items-center p-5  ">
          <h1 className="text-4xl font-bold m-3">Login </h1>

          <form className="w-full" onSubmit={handleSubmit}>
            <div className="m-3">
              <label className="text-lg" htmlFor="email">
                Email
              </label>
              <br />
              <input
                type="text"
                id="email"
                name="email"
                className="mt-1 rounded-lg shadow-lg border p-3 w-full border-gray-300 "
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>
            <br />
            <div className="m-3">
              <label htmlFor="password" className=" text-lg">
                Password
              </label>
              <br />
              <input
                type="password"
                id="password"
                name="password"
                className=" mt-1 rounded-lg shadow-lg border p-3 w-full border-gray-300"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
            </div>
            <div className="flex justify-center m-8 ">
              <button
                type="submit"
                className="bg-blue-500 rounded-lg text-white p-2 text-lg hover:bg-blue-600 hover:shadow-lg w-1/5"
              >
                Login
              </button>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
}
