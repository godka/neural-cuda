#include "libconvert.h"

std::string readStringFromFile(const std::string &filename)
{
	FILE *fp = fopen(filename.c_str(), "rb");
	if (!fp)
	{
		printf("Can not open file %s\n", filename.c_str());
		return "";
	}
	fseek(fp, 0, SEEK_END);
	int length = ftell(fp);
	fseek(fp, 0, 0);
	char* s = new char[length + 1];
	for (int i = 0; i <= length; s[i++] = '\0');
	fread(s, length, 1, fp);
	std::string str(s);
	fclose(fp);
	delete[] s;
	return str;
}

void writeStringToFile(const std::string &str, const std::string &filename)
{
	FILE *fp = fopen(filename.c_str(), "wb");
	int length = str.length();
	fwrite(str.c_str(), length, 1, fp);
	fclose(fp);
}

int replaceString(std::string &s, const std::string &oldstring, const std::string &newstring, int pos0/*=0*/)
{
	int pos = s.find(oldstring, pos0);
	if (pos >= 0)
	{
		s.erase(pos, oldstring.length());
		s.insert(pos, newstring);
	}
	return pos + newstring.length();
}

int replaceAllString(std::string &s, const std::string &oldstring, const std::string &newstring)
{
	int pos = s.find(oldstring);
	while (pos >= 0)
	{
		s.erase(pos, oldstring.length());
		s.insert(pos, newstring);
		pos = s.find(oldstring, pos+ newstring.length());
	}
	return pos + newstring.length();
}

void replaceStringInFile(const std::string &oldfilename, const std::string &newfilename, const std::string &oldstring, const std::string &newstring)
{
	std::string s = readStringFromFile(oldfilename);
	if (s.length() <= 0) return;
	replaceString(s, oldstring, newstring);
	writeStringToFile(s, newfilename);
}

void replaceAllStringInFile(const std::string &oldfilename, const std::string &newfilename, const std::string &oldstring, const std::string &newstring)
{
	std::string s = readStringFromFile(oldfilename);
	if (s.length() <= 0) return;
	replaceAllString(s, oldstring, newstring);
	writeStringToFile(s, newfilename);
}

std::string formatString(const char *format, ...)
{
	char s[1000];
	va_list arg_ptr;
	va_start(arg_ptr, format);
	vsprintf(s, format, arg_ptr);
	va_end(arg_ptr);
	return s;
}

void formatAppendString(std::string &str, const char *format, ...)
{
	char s[1000];
	va_list arg_ptr;
	va_start(arg_ptr, format);
	vsprintf(s, format, arg_ptr);
	va_end(arg_ptr);
	str += s;
}

double diff1(double y1, double x1, double y0, double x0)
{
	return (y1 - y0) / (x1 - x0);
}

double diff2(double y2, double x2, double y1, double x1, double y0, double x0)
{
	return (diff1(y2, x2, y1, x1) - diff1(y1, x1, y0, x0)) / (x1 - x0);
}

int findNumbers(const std::string &s, std::vector<double> &data)
{
	int n = 0;
	std::string str = "";
	bool haveNum = false;
	for (int i = 0; i < s.length(); i++)
	{
		char c = s[i];
		if ((c >= '0' && c <= '9') || c == '.' || c == '-' || c == '+' || c == 'E' || c == 'e')
		{
			str += c;
			if (c >= '0' && c <= '9')
				haveNum = true;
		}
		else
		{
			if (str != "" && haveNum)
			{
				double f = atof(str.c_str());
				data.push_back(f);
				n++;
			}
			str = "";
			haveNum = false;
		}
	}
	return n;
}

std::string findANumber(const std::string &s)
{
	bool findPoint = false;
	bool findNumber = false;
	bool findE = false;
	std::string n;
	for (int i = 0; i < s.length(); i++)
	{
		char c = s[i];
		if (c >= '0' && c <= '9' || c=='-' || c == '.' || c=='e' || c=='E')
		{
			if (c >= '0' && c <= '9' || c == '-')
			{
				findNumber = true;
				n += c;
			}
			if (c == '.')
			{
				if (!findPoint)
					n += c;
				findPoint = true;
			}
			if (c == 'e' || c == 'E')
			{
				if (findNumber && !(findE))
				{
					n += c;
					findE = true;
				}
			}
		}
		else
		{
			if (findNumber)
				break;
		}
	}
	return n;
}

unsigned findTheLast(const std::string &s, const std::string &content)
{
	int pos = 0, prepos = 0;
	while (pos >= 0)
	{
		prepos = pos;
		pos = s.find(content, prepos + 1);
		//printf("%d\n",pos);
	}
	return prepos;
}

std::vector<std::string> splitString(std::string str, std::string pattern)
{
	std::string::size_type pos;
	std::vector<std::string> result;
	str += pattern; //��չ�ַ����Է������
	int size = str.size();

	for (int i = 0; i < size; i++)
	{
		pos = str.find(pattern, i);
		if (pos < size)
		{
			std::string s = str.substr(i, pos - i);
			result.push_back(s);
			i = pos + pattern.size() - 1;
		}
	}
	return result;
}

bool isProChar(char c)
{
	return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'z') || (c >= '(' && c <= ')');
}