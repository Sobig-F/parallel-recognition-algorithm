#ifndef WIN_GUI_HPP
#define WIN_GUI_HPP

#include <filesystem>

namespace biometrics
{
int RunGuiApp(const std::filesystem::path& modelPath, const std::filesystem::path& databaseRoot);
} // namespace biometrics

#endif
