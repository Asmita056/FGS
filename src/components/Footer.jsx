import "@fortawesome/fontawesome-free/css/all.min.css";

export default function Footer() {
  return (
    <>
      <div className="mb-6 bg-lime-400">
        <div className="text-center mt-4 mb-3">
          <h2 className="text-[#2E8D49]">
            Stay connected and follow our journey:
          </h2>
        </div>
        <div className="container mx-auto flex justify-center space-x-6">
          <a
            href="https://facebook.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-600 text-xl"
          >
            <i className="fab fa-facebook-f"></i>
          </a>
          <a
            href="https://twitter.com"
            target="_blank"
            rel="noopener noreferrer"
            className=" text-blue-400 text-xl"
          >
            <i className="fab fa-twitter"></i>
          </a>
          <a
            href="https://instagram.com"
            target="_blank"
            rel="noopener noreferrer"
            className=" text-pink-500 text-xl"
          >
            <i className="fab fa-instagram"></i>
          </a>
          <a
            href="https://linkedin.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-700 text-xl"
          >
            <i className="fab fa-linkedin-in"></i>
          </a>
          <a
            href="https://youtube.com"
            target="_blank"
            rel="noopener noreferrer"
            className="text-red-600 text-xl"
          >
            <i className="fab fa-youtube"></i>
          </a>
        </div>
      </div>
    </>
  );
}
