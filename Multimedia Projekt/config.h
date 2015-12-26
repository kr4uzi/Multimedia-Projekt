#pragma once
#include <string>
#include <map>

namespace mmp
{
	class config
	{
	public:
		class parse_error
		{
		private:
			std::string error_message;

		public:
			parse_error(const std::string& error_message)
				: error_message(error_message)
			{ }

			std::string error_msg() const { return error_message; }
			bool operator!() const { return error_message.empty(); }
		};

	private:
		std::map<std::string, std::string> cfg;

	public:
		static parse_error parse(const std::string& filename, config& cfg);

		bool exists(const std::string& key) const;

		std::string get(const std::string& key, const std::string& default_value = "") const;
		double get(const std::string& key, double default_value = 0) const;
		unsigned get(const std::string& key, unsigned default_value = 0) const;
	};
}