import React from "react";
import { useNavigate } from "react-router-dom";

export default function Profile() {
  const navigate = useNavigate();

  const handleLogout = () => {
    // e.preventDefault();

    navigate("/", { replace: true });
  };
  return (
    <>
      <div>Profile</div>
      <button
        onClick={handleLogout}
        className="bg-blue-500 rounded-lg text-white p-2 text-lg hover:bg-blue-600 hover:shadow-lg w-1/5"
      >
        Logout
      </button>
    </>
  );
}
