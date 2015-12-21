#pragma once

#ifndef _MMC_COMMON_DATATYPE_CONVERTER
#define _MMC_COMMON_DATATYPE_CONVERTER

#include <sstream>

using namespace std;

namespace mmc_common
{
	class DatatypeConverter
	{
	public:
		static inline string convertIntToString(const int& val)
		{
			return convertNumericTypeToString(val);
		}

		static inline string convertInt64ToString(const __int64& val)
		{
			return convertNumericTypeToString(val);
		}

		static inline string convertFloatToString(const float& val)
		{
			return convertNumericTypeToString(val);
		}

		static inline string convertDoubleToString(const double& val)
		{
			return convertNumericTypeToString(val);
		}

		static inline string convertBoolToString(const bool& val)
		{
			return (val == true) ? "true" : "false";
		}

	private:
		template <class T>
		static inline string convertNumericTypeToString(const T& t)
		{
			std::stringstream ss;
			ss << t;

			return ss.str();
		}
	};
}

#endif