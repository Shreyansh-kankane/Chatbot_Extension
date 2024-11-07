import { NavLink } from "react-router-dom";

export default function Navbar() {
  return (
    <div>
      <nav className="flex flex-row  items-center w-full bg-[#16141A] p-4 shadow-md text-white">
        <div className="flex flex-row w-1/2 text-xl font-bold">Logo</div>
        <div className="flex flex-row w-1/2 justify-end text-xl font-bold">
          <NavLink to="/onboarding/login" className="p-2 mx-5">
            Login
          </NavLink>
          <NavLink to="/onboarding/register" className="p-2 mx-5">
            Register
          </NavLink>
        </div>
      </nav>
    </div>
  );
}
