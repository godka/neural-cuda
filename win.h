#pragma once

#ifdef _MSC_VER
#include <stdio.h>
#include <windows.h>
#endif

void set_console_color(unsigned short color_index)
{
#ifdef _MSC_VER
	SetConsoleTextAttribute(GetStdHandle(STD_OUTPUT_HANDLE), color_index);
#endif
}
