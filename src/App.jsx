import { useState } from "react";
import {
  BrowserRouter as Router,
  Routes,
  Route,
  useLocation,
} from "react-router-dom";
import NavBar from "./components/NavBar";
import MainPage from "./components/MainPage";
import Analysis from "./components/Analysis";
import AboutUs from "./components/AboutUs";
import ContactUs from "./components/ContactUs";
import Profile from "./components/Profile";
import Footer from "./components/Footer";
import SignIn from "./components/SignIn";

function Layout({ email }) {
  const location = useLocation();
  const hideHeaderFooter = location.pathname === "/";

  return (
    <>
      {!hideHeaderFooter && <NavBar email={email} />}
      <Routes>
        <Route path="/Home" element={<MainPage />} />
        <Route path="/Analysis" element={<Analysis />} />
        <Route path="/AboutUs" element={<AboutUs />} />
        <Route path="/ContactUs" element={<ContactUs />} />
        <Route path="/Profile" element={<Profile />} />
        <Route path="/" element={<SignIn />} />
      </Routes>
      {!hideHeaderFooter && <Footer />}
    </>
  );
}

function App() {
  const [email, setEmail] = useState("123@gmail.com");
  return (
    <Router>
      <Layout email={email} />
    </Router>
  );
}

export default App;
