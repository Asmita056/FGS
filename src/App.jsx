import { useState } from "react";
import { BrowserRouter as Router, Route, Routes } from "react-router-dom";
import NavBar from "./components/NavBar";
import MainPage from "./components/MainPage";
import Analysis from "./components/Analysis";
import AboutUs from "./components/AboutUs";
import ContactUs from "./components/ContactUs";
import Profile from "./components/Profile";
import Footer from "./components/Footer";

function App() {
  return (
    <>
      <Router>
        <NavBar />
        <Routes>
          <Route path="/" element={<MainPage />} />
          <Route path="/Analysis" element={<Analysis />} />
          <Route path="/AboutUs" element={<AboutUs />} />
          <Route path="/ContactUs" element={<ContactUs />} />
          <Route path="/Profile" element={<Profile />} />
        </Routes>
        <Footer />
      </Router>
    </>
  );
}

export default App;
