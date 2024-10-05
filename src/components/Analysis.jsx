// import totalCount from "./MainPage"

export default function Analysis() {
  return (
    <div className="bg-[#F3FFCF] p-3">
      <div>Welcome to your Analytics!</div>

      <div className="flex justify-between m-2">
        <div className="flex border rounded-lg inline-flex ">
          <button
            type="submit"
            className="border-gray-500  p-1 hover:bg-lime-300 rounded-md"
          >
            Daily
          </button>
          <button
            type="submit"
            className="border-gray-500  p-1 hover:bg-lime-300 rounded-md"
          >
            weekly
          </button>
          <button
            type="submit"
            className="border-gray-500  p-1 hover:bg-lime-300 rounded-md"
          >
            Monthly
          </button>
        </div>
      </div>

      <div className="flex justify-between">
        <div className="bg-lime-300 p-4 m-4">
          <div>Total Fruits Graded</div>
          <div>{}</div>
        </div>
        <div className="bg-lime-300 p-4 m-4">
          <div>Best count</div>
          <div>{}</div>
        </div>
        <div className="bg-lime-300 p-4 m-4">
          <div>Average count</div>
          <div>{}</div>
        </div>
        <div className="bg-lime-300 p-4 m-4">
          <div>worst count</div>
          <div>{}</div>
        </div>
      </div>
    </div>
  );
}
