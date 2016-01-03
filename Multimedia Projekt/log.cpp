#include "log.h"
#include "helpers.h"
using namespace mmp;

logger::logger(const std::string& filename)
	: target(to::both)
{
	log_file.open(filename, std::ios::app);
	if (log_file.is_open())
	{
		log_file << time_string() << " ";
	}
	else
		std::cout << "could not open logfile [" << filename << "]" << std::endl;
}

logger::~logger()
{
	if (log_file.is_open())
	{
		log_file << std::endl;
		log_file.close();
	}
}

logger mmp::log("mmp.log");