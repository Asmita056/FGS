import { Link } from 'react-router-dom';

export default function NavBar() {
    return (

        <>
            <div className="bg-green-700 text-white  flex justify-between px-4 py-2 mb-5 shadow-xl ">
                <div className=" text-3xl">
                    Fruit Grading System
                </div>
                <div>
                    <ul className="flex justify-end ">
                        <li className="p-3"><Link to="/">Home</Link></li>
                        <li className="p-3"><Link to="/Analysis">Analysis</Link></li>
                        <li className="p-3"><Link to="/AboutUs">About Us</Link></li>
                        <li className="p-3"><Link to="/ContactUs">Contact Us</Link></li>
                        <li className="p-3"><Link to="/Profile">Profile</Link></li>
                    </ul>
                </div>
            </div>
        </>
    )
}