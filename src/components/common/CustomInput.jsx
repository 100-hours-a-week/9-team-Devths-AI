export function CustomInput({type, name, placeholder, onChange, required}) {
   return (
      <input 
         type={type}
         name={name}
         placeholder={placeholder}
         value={formData.groomName}
         onChange={onChange}
                  className="w-full px-5 py-4 border-2 border-gray-200 rounded-xl focus:border-rose-400 focus:outline-none transition-colors text-lg"
                  required>

      </input>
   );
}