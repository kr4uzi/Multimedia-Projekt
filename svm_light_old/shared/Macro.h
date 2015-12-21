
#ifndef _MMC_COMMON_MACRO
#define _MMC_COMMON_MACRO

#include <iostream>
#include <string>
#include <sstream>
#include <stdio.h>
#include <stdlib.h>

#ifndef _check
	/**
	* Checks if \a token is true. 
	* If NOT, a message is logged, the debugger is triggered,
	* an exception is thrown and the program then quits.
	*
	* This check is always active.
	*/
	#define _check(token, ...) {															\
		if(!(token)){																		\
			std::cerr << "\nAn error occurred at "<<__FILE__<<"("<<__LINE__<<")@"<<__FUNCTION__<<"(),"<< std::endl;	\
			std::cerr << "Condition failed: _check(" #token "). "  __VA_ARGS__ << std::endl;	\
			system("pause");																\
			abort();																		\
		}																					\
	}
#endif

// Creates at runtime a string from an expression like ' " error: " << a.getNumTotal() << " is invalid." '
#define __iEXPRESSION_TO_STREAM(expression)	std::stringstream __tempStrStr;			\
											__tempStrStr.precision(14);				\
											__tempStrStr << expression;				\
											std::string __tmpStr = __tempStrStr.str();

// Creates at runtime a string from an expression like ' " error: %s, %d", ex.what(), 123 '
#ifdef _MSC_VER
	#define __iEXPRESSION_TO_PRINTF2(formatString, len, ...)	char __tmpStrBuf[1024]; \
														sprintf_s(__tmpStrBuf, 1024, formatString, __VA_ARGS__);
#else
	#define __iEXPRESSION_TO_PRINTF2(formatString, len, ...)	char __tmpStrBuf[len];  \
														sprintf(__tmpStrBuf, formatString, __VA_ARGS__);
#endif


/**
 * Logs a message with level TRACE.
 * Usage:
 * <pre>
 *		_trace("Operation done in " << t << " seconds.");
 *		_trace("Operation done in %d seconds.", t);
 * </pre>
 */
#define log_message(expression, ...)	{															\
	__iEXPRESSION_TO_STREAM("\n"<<__FILE__<<"("<<__LINE__<<")@"<<__FUNCTION__<<"():\n  "<< expression);\
	__iEXPRESSION_TO_PRINTF2(__tmpStr.c_str(), __tmpStr.length(), __VA_ARGS__);				\
	std::cout << __tmpStrBuf << std::endl;													\
}

#define log_message_exit(expression, ...)	{															\
	__iEXPRESSION_TO_STREAM("\n"<<__FILE__<<"("<<__LINE__<<")@"<<__FUNCTION__<<"():\n  "<< expression);\
	__iEXPRESSION_TO_PRINTF2(__tmpStr.c_str(), __tmpStr.length(), __VA_ARGS__);				\
	std::cout << __tmpStrBuf << std::endl;													\
	exit(EXIT_FAILURE); \
}

#define cnd_printf(condition, expression, ...)	{															\
	if(condition) { \
	__iEXPRESSION_TO_STREAM(expression);\
	__iEXPRESSION_TO_PRINTF2(__tmpStr.c_str(), __tmpStr.length(), __VA_ARGS__);				\
	std::cout << __tmpStrBuf;			\
	}\
}

#define _trace log_message
#define _debug log_message
#define _info log_message
#define _warn log_message
#define _error log_message_exit

#endif