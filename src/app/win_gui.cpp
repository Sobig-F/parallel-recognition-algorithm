#include "app/win_gui.hpp"

#define NOMINMAX
#include <Windows.h>
#include <commctrl.h>
#include <commdlg.h>
#include <shlobj.h>

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cctype>
#include <cstdlib>
#include <ctime>
#include <exception>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <locale>
#include <memory>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "app/recognition_engine.hpp"

namespace biometrics
{
namespace
{
constexpr int kNameEditId = 1001;
constexpr int kAddButtonId = 1002;
constexpr int kSearchButtonId = 1003;
constexpr int kReloadButtonId = 1004;
constexpr int kCustomThresholdEditId = 1005;
constexpr int kOpenCvThresholdEditId = 1006;
constexpr int kCustomCpuCheckId = 1007;
constexpr int kCustomCudaCheckId = 1008;
constexpr int kOpenCvCpuCheckId = 1009;
constexpr int kOpenCvCudaCheckId = 1010;

constexpr UINT kBusyMessage = WM_APP + 1;
constexpr UINT kReportMessage = WM_APP + 2;
constexpr UINT kStatusMessage = WM_APP + 3;

constexpr COLORREF kBackground = RGB(244, 247, 251);
constexpr COLORREF kSurface = RGB(255, 255, 255);
constexpr COLORREF kText = RGB(23, 28, 39);
constexpr COLORREF kMuted = RGB(102, 112, 133);
constexpr COLORREF kAccent = RGB(37, 99, 235);

struct ReportPayload
{
    SearchReport report;
    std::filesystem::path csvPath;
};

std::wstring Utf8ToWide(const std::string& value)
{
    if (value.empty())
    {
        return {};
    }

    int size = MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0);
    if (size <= 0)
    {
        return {};
    }

    std::wstring result(size, L'\0');
    MultiByteToWideChar(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), size);
    return result;
}

std::string WideToUtf8(const std::wstring& value)
{
    if (value.empty())
    {
        return {};
    }

    int size = WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), nullptr, 0, nullptr, nullptr);
    if (size <= 0)
    {
        return {};
    }

    std::string result(size, '\0');
    WideCharToMultiByte(CP_UTF8, 0, value.data(), static_cast<int>(value.size()), result.data(), size, nullptr, nullptr);
    return result;
}

std::wstring GetWindowTextString(HWND window)
{
    int size = GetWindowTextLengthW(window);
    std::wstring text(size + 1, L'\0');
    int copied = GetWindowTextW(window, text.data(), size + 1);
    text.resize(copied);
    return text;
}

std::string FormatDouble(double value, int precision = 2)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string TimestampForFile()
{
    auto now = std::chrono::system_clock::now();
    std::time_t time = std::chrono::system_clock::to_time_t(now);
    std::tm local{};
    localtime_s(&local, &time);

    std::ostringstream out;
    out << std::put_time(&local, "%Y%m%d_%H%M%S");
    return out.str();
}

std::vector<std::filesystem::path> SelectImages(HWND owner)
{
    std::vector<wchar_t> buffer(65536, L'\0');
    OPENFILENAMEW ofn{};
    ofn.lStructSize = sizeof(ofn);
    ofn.hwndOwner = owner;
    ofn.lpstrFile = buffer.data();
    ofn.nMaxFile = static_cast<DWORD>(buffer.size());
    ofn.lpstrFilter =
        L"Изображения\0*.jpg;*.jpeg;*.png;*.bmp;*.webp;*.tif;*.tiff\0"
        L"Все файлы\0*.*\0";
    ofn.nFilterIndex = 1;
    ofn.Flags = OFN_EXPLORER | OFN_ALLOWMULTISELECT | OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST;

    if (!GetOpenFileNameW(&ofn))
    {
        return {};
    }

    std::vector<std::filesystem::path> paths;
    wchar_t* cursor = buffer.data();
    std::wstring first = cursor;
    cursor += first.size() + 1;

    if (*cursor == L'\0')
    {
        paths.emplace_back(first);
        return paths;
    }

    std::filesystem::path directory(first);
    while (*cursor != L'\0')
    {
        std::wstring fileName = cursor;
        paths.push_back(directory / fileName);
        cursor += fileName.size() + 1;
    }

    return paths;
}

std::filesystem::path SelectFolder(HWND owner)
{
    BROWSEINFOW browse{};
    browse.hwndOwner = owner;
    browse.lpszTitle = L"Выберите папку с фотографиями для проверки";
    browse.ulFlags = BIF_RETURNONLYFSDIRS | BIF_NEWDIALOGSTYLE;

    PIDLIST_ABSOLUTE pidl = SHBrowseForFolderW(&browse);
    if (pidl == nullptr)
    {
        return {};
    }

    wchar_t path[MAX_PATH]{};
    BOOL ok = SHGetPathFromIDListW(pidl, path);
    CoTaskMemFree(pidl);
    if (!ok)
    {
        return {};
    }

    return std::filesystem::path(path);
}

HMENU ControlId(int id)
{
    return reinterpret_cast<HMENU>(static_cast<INT_PTR>(id));
}

void AddColumn(HWND list, int index, int width, const wchar_t* text)
{
    LVCOLUMNW column{};
    column.mask = LVCF_TEXT | LVCF_WIDTH | LVCF_SUBITEM;
    column.cx = width;
    column.pszText = const_cast<wchar_t*>(text);
    column.iSubItem = index;
    SendMessageW(list, LVM_INSERTCOLUMNW, index, reinterpret_cast<LPARAM>(&column));
}

int AddRow(HWND list, const std::vector<std::wstring>& values)
{
    LVITEMW item{};
    item.mask = LVIF_TEXT;
    item.iItem = ListView_GetItemCount(list);
    item.pszText = const_cast<wchar_t*>(values.empty() ? L"" : values[0].c_str());
    int row = static_cast<int>(SendMessageW(list, LVM_INSERTITEMW, 0, reinterpret_cast<LPARAM>(&item)));

    for (int column = 1; column < static_cast<int>(values.size()); ++column)
    {
        LVITEMW subItem{};
        subItem.iSubItem = column;
        subItem.pszText = const_cast<wchar_t*>(values[column].c_str());
        SendMessageW(list, LVM_SETITEMTEXTW, row, reinterpret_cast<LPARAM>(&subItem));
    }

    return row;
}

void SetText(HWND control, const std::string& text)
{
    SetWindowTextW(control, Utf8ToWide(text).c_str());
}

double ParseDoubleFromEdit(HWND edit, double fallback)
{
    std::string text = WideToUtf8(GetWindowTextString(edit));
    text.erase(std::remove_if(text.begin(), text.end(), [](unsigned char ch) {
        return std::isspace(ch) != 0;
    }), text.end());
    std::replace(text.begin(), text.end(), ',', '.');

    if (text.empty())
    {
        return fallback;
    }

    double value = fallback;
    auto [ptr, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error != std::errc{} || ptr != text.data() + text.size() || value < 0.0 || value > 1.0)
    {
        return fallback;
    }

    return value;
}

std::string VerdictText(bool recognized)
{
    return recognized ? "Пропуск" : "Отказ";
}

std::string OutcomeText(const ImageAlgorithmResult& result)
{
    if (!result.error.empty())
    {
        return "Сбой";
    }
    return result.correct ? "Успех" : "Ошибка";
}

std::string ErrorText(const ImageAlgorithmResult& result)
{
    if (!result.error.empty())
    {
        return result.error;
    }
    if (result.errorType == "FALSE_ACCEPT" || result.errorType == "ЛОЖНЫЙ_ПРОПУСК")
    {
        return "Ложный пропуск";
    }
    if (result.errorType == "FALSE_REJECT" || result.errorType == "ЛОЖНЫЙ_ОТКАЗ")
    {
        return "Ложный отказ";
    }
    if (result.errorType == "WRONG_MATCH" || result.errorType == "НЕ_ТО_ЛИЦО")
    {
        return "Неверное лицо";
    }
    return {};
}

bool ContainsAlgorithm(const std::vector<AlgorithmId>& algorithms, AlgorithmId id)
{
    return std::find(algorithms.begin(), algorithms.end(), id) != algorithms.end();
}

double ClassRecall(int correct, int total)
{
    return total > 0 ? static_cast<double>(correct) / static_cast<double>(total) : 0.0;
}

double BalancedAccuracyPercent(const AlgorithmRunSummary& summary)
{
    const bool hasPassClass = summary.expectedPass > 0;
    const bool hasDenyClass = summary.expectedDeny > 0;
    const double passRecall = ClassRecall(summary.correctPasses, summary.expectedPass);
    const double denyRecall = ClassRecall(summary.correctDenies, summary.expectedDeny);

    if (hasPassClass && hasDenyClass)
    {
        return 100.0 * (passRecall + denyRecall) / 2.0;
    }
    if (hasPassClass)
    {
        return 100.0 * passRecall;
    }
    if (hasDenyClass)
    {
        return 100.0 * denyRecall;
    }
    return 0.0;
}

class MainWindow
{
public:
    MainWindow(std::filesystem::path modelPath, std::filesystem::path databaseRoot)
        : _engine(std::make_unique<RecognitionEngine>(std::move(modelPath), std::move(databaseRoot)))
    {
    }

    int Run()
    {
        CreateBrushes();

        INITCOMMONCONTROLSEX controls{};
        controls.dwSize = sizeof(controls);
        controls.dwICC = ICC_LISTVIEW_CLASSES | ICC_STANDARD_CLASSES;
        InitCommonControlsEx(&controls);

        HINSTANCE instance = GetModuleHandleW(nullptr);
        WNDCLASSW windowClass{};
        windowClass.lpfnWndProc = &MainWindow::WindowProc;
        windowClass.hInstance = instance;
        windowClass.lpszClassName = L"BiometricsFaceControlWindow";
        windowClass.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
        windowClass.hbrBackground = _backgroundBrush;
        RegisterClassW(&windowClass);

        _window = CreateWindowExW(
            0,
            windowClass.lpszClassName,
            L"Фейс-контроль LBPH",
            WS_OVERLAPPEDWINDOW,
            CW_USEDEFAULT,
            CW_USEDEFAULT,
            1320,
            860,
            nullptr,
            nullptr,
            instance,
            this
        );

        if (_window == nullptr)
        {
            return 1;
        }

        ShowWindow(_window, SW_SHOW);
        UpdateWindow(_window);

        MSG message{};
        while (GetMessageW(&message, nullptr, 0, 0))
        {
            TranslateMessage(&message);
            DispatchMessageW(&message);
        }

        return static_cast<int>(message.wParam);
    }

private:
    static LRESULT CALLBACK WindowProc(HWND window, UINT message, WPARAM wParam, LPARAM lParam)
    {
        MainWindow* self = nullptr;
        if (message == WM_NCCREATE)
        {
            auto* create = reinterpret_cast<CREATESTRUCTW*>(lParam);
            self = static_cast<MainWindow*>(create->lpCreateParams);
            SetWindowLongPtrW(window, GWLP_USERDATA, reinterpret_cast<LONG_PTR>(self));
            self->_window = window;
        }
        else
        {
            self = reinterpret_cast<MainWindow*>(GetWindowLongPtrW(window, GWLP_USERDATA));
        }

        if (self == nullptr)
        {
            return DefWindowProcW(window, message, wParam, lParam);
        }

        return self->HandleMessage(message, wParam, lParam);
    }

    LRESULT HandleMessage(UINT message, WPARAM wParam, LPARAM lParam)
    {
        switch (message)
        {
        case WM_CREATE:
            CreateControls();
            RefreshMetrics();
            SeedAlgorithmTable("ожидание", SelectedAlgorithmsFromUi());
            PostStatus("Готово");
            PostLog("Приложение запущено.");
            PostLog(IsCudaAvailable() ? "CUDA доступна." : "CUDA недоступна, GPU-алгоритмы будут пропущены.");
            return 0;
        case WM_SIZE:
            LayoutControls(LOWORD(lParam), HIWORD(lParam));
            return 0;
        case WM_GETMINMAXINFO: {
            auto* info = reinterpret_cast<MINMAXINFO*>(lParam);
            info->ptMinTrackSize.x = 1240;
            info->ptMinTrackSize.y = 780;
            return 0;
        }
        case WM_ERASEBKGND:
            PaintBackground(reinterpret_cast<HDC>(wParam));
            return 1;
        case WM_CTLCOLORSTATIC:
            return HandleControlColor(reinterpret_cast<HDC>(wParam), reinterpret_cast<HWND>(lParam));
        case WM_CTLCOLOREDIT:
            SetTextColor(reinterpret_cast<HDC>(wParam), kText);
            SetBkColor(reinterpret_cast<HDC>(wParam), kSurface);
            return reinterpret_cast<LRESULT>(_surfaceBrush);
        case WM_COMMAND:
            HandleCommand(LOWORD(wParam));
            return 0;
        case kStatusMessage:
            SetWindowTextW(_statusText, reinterpret_cast<std::wstring*>(lParam)->c_str());
            delete reinterpret_cast<std::wstring*>(lParam);
            return 0;
        case kReportMessage:
            RenderReport(*reinterpret_cast<ReportPayload*>(lParam));
            delete reinterpret_cast<ReportPayload*>(lParam);
            return 0;
        case kBusyMessage:
            SetBusy(wParam != 0);
            return 0;
        case WM_DESTROY:
            DeleteBrushes();
            DeleteFonts();
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(_window, message, wParam, lParam);
        }
    }

    void CreateBrushes()
    {
        _backgroundBrush = CreateSolidBrush(kBackground);
        _surfaceBrush = CreateSolidBrush(kSurface);
        _accentBrush = CreateSolidBrush(kAccent);
    }

    void DeleteBrushes()
    {
        for (HBRUSH brush : {_backgroundBrush, _surfaceBrush, _accentBrush})
        {
            if (brush != nullptr)
            {
                DeleteObject(brush);
            }
        }
    }

    void DeleteFonts()
    {
        for (HFONT font : {_titleFont, _sectionFont, _font, _smallFont, _badgeFont})
        {
            if (font != nullptr)
            {
                DeleteObject(font);
            }
        }
    }

    void PaintBackground(HDC dc)
    {
        RECT rect{};
        GetClientRect(_window, &rect);
        FillRect(dc, &rect, _backgroundBrush);
    }

    LRESULT HandleControlColor(HDC dc, HWND control)
    {
        SetBkMode(dc, TRANSPARENT);
        SetTextColor(dc, kText);

        if (control == _statusText)
        {
            SetTextColor(dc, RGB(255, 255, 255));
            SetBkMode(dc, OPAQUE);
            SetBkColor(dc, kAccent);
            return reinterpret_cast<LRESULT>(_accentBrush);
        }

        if (control == _metricDb || control == _metricCuda ||
            control == _actionPanel || control == _summaryPanel || control == _resultsPanel ||
            control == _actionTitle || control == _nameLabel ||
            control == _customThresholdLabel || control == _openCvThresholdLabel ||
            control == _algorithmLabel || control == _customCpuCheck || control == _customCudaCheck ||
            control == _openCvCpuCheck || control == _openCvCudaCheck ||
            control == _hintText || control == _lastReport ||
            control == _summaryTitle || control == _resultsTitle)
        {
            if (control == _hintText || control == _lastReport)
            {
                SetTextColor(dc, kMuted);
            }
            SetBkMode(dc, OPAQUE);
            SetBkColor(dc, kSurface);
            return reinterpret_cast<LRESULT>(_surfaceBrush);
        }

        if (control == _subtitle)
        {
            SetTextColor(dc, kMuted);
        }

        return reinterpret_cast<LRESULT>(_backgroundBrush);
    }

    HWND CreateLabel(const wchar_t* text)
    {
        return CreateWindowExW(0, L"STATIC", text, WS_CHILD | WS_VISIBLE,
                               0, 0, 10, 10, _window, nullptr, nullptr, nullptr);
    }

    HWND CreatePanel()
    {
        return CreateWindowExW(WS_EX_CONTROLPARENT, L"STATIC", L"", WS_CHILD | WS_VISIBLE,
                               0, 0, 10, 10, _window, nullptr, nullptr, nullptr);
    }

    HWND CreateButton(const wchar_t* text, int id)
    {
        return CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | BS_PUSHBUTTON,
                               0, 0, 10, 10, _window, ControlId(id), nullptr, nullptr);
    }

    HWND CreateEdit(const wchar_t* text, int id)
    {
        return CreateWindowExW(WS_EX_CLIENTEDGE, L"EDIT", text,
                               WS_CHILD | WS_VISIBLE | ES_AUTOHSCROLL,
                               0, 0, 10, 10, _window, ControlId(id), nullptr, nullptr);
    }

    HWND CreateCheckbox(const wchar_t* text, int id)
    {
        return CreateWindowExW(0, L"BUTTON", text, WS_CHILD | WS_VISIBLE | BS_AUTOCHECKBOX,
                               0, 0, 10, 10, _window, ControlId(id), nullptr, nullptr);
    }

    void CreateControls()
    {
        _titleFont = CreateFontW(30, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                 OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
        _sectionFont = CreateFontW(19, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                   OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                   DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
        _font = CreateFontW(16, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                            OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                            DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
        _smallFont = CreateFontW(14, 0, 0, 0, FW_NORMAL, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                 OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");
        _badgeFont = CreateFontW(15, 0, 0, 0, FW_SEMIBOLD, FALSE, FALSE, FALSE, DEFAULT_CHARSET,
                                 OUT_DEFAULT_PRECIS, CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY,
                                 DEFAULT_PITCH | FF_DONTCARE, L"Segoe UI");

        _title = CreateLabel(L"Фейс-контроль LBPH");
        _subtitle = CreateLabel(L"Проверка лиц по базе: пропуск или отказ, с оценкой точности каждого алгоритма.");
        _statusText = CreateLabel(L"Готово");

        _metricDb = CreateLabel(L"Лиц в базе\n0");
        _metricCuda = CreateLabel(L"CUDA\n—");

        _actionPanel = CreatePanel();
        _actionTitle = CreateLabel(L"Управление проверкой");
        _nameLabel = CreateLabel(L"Имя нового лица");
        _nameEdit = CreateEdit(L"", kNameEditId);
        _addButton = CreateButton(L"Добавить лицо", kAddButtonId);
        _searchButton = CreateButton(L"Проверить папку", kSearchButtonId);
        _reloadButton = CreateButton(L"Обновить базу", kReloadButtonId);
        _customThresholdLabel = CreateLabel(L"Порог моей LBPH");
        _customThresholdEdit = CreateEdit(L"0.95", kCustomThresholdEditId);
        _openCvThresholdLabel = CreateLabel(L"Порог OpenCV LBPH");
        _openCvThresholdEdit = CreateEdit(L"0.03", kOpenCvThresholdEditId);
        _algorithmLabel = CreateLabel(L"Алгоритмы для запуска");
        _customCpuCheck = CreateCheckbox(L"Моя LBPH CPU", kCustomCpuCheckId);
        _customCudaCheck = CreateCheckbox(L"Моя LBPH CUDA", kCustomCudaCheckId);
        _openCvCpuCheck = CreateCheckbox(L"OpenCV LBPH CPU", kOpenCvCpuCheckId);
        _openCvCudaCheck = CreateCheckbox(L"OpenCV LBPH CUDA", kOpenCvCudaCheckId);
        _hintText = CreateLabel(L"Ожидаемое имя берется из файла: Ivan Petrov_12.jpg -> Ivan Petrov");
        _lastReport = CreateLabel(L"CSV-отчет пока не создан");

        SendMessageW(_customCpuCheck, BM_SETCHECK, BST_CHECKED, 0);
        SendMessageW(_openCvCpuCheck, BM_SETCHECK, BST_CHECKED, 0);
        SendMessageW(_customCudaCheck, BM_SETCHECK, IsCudaAvailable() ? BST_CHECKED : BST_UNCHECKED, 0);
        SendMessageW(_openCvCudaCheck, BM_SETCHECK, IsCudaAvailable() ? BST_CHECKED : BST_UNCHECKED, 0);
        EnableWindow(_customCudaCheck, IsCudaAvailable());
        EnableWindow(_openCvCudaCheck, IsCudaAvailable());

        _summaryPanel = CreatePanel();
        _summaryTitle = CreateLabel(L"Статистика по алгоритмам");
        _summaryList = CreateWindowExW(WS_EX_CLIENTEDGE, WC_LISTVIEWW, L"",
                                       WS_CHILD | WS_VISIBLE | LVS_REPORT | LVS_SINGLESEL,
                                       0, 0, 10, 10, _window, nullptr, nullptr, nullptr);

        _resultsPanel = CreatePanel();
        _resultsTitle = CreateLabel(L"Результаты по файлам");
        _resultsList = CreateWindowExW(WS_EX_CLIENTEDGE, WC_LISTVIEWW, L"",
                                       WS_CHILD | WS_VISIBLE | LVS_REPORT | LVS_SINGLESEL,
                                       0, 0, 10, 10, _window, nullptr, nullptr, nullptr);

        ApplyFonts();
        ConfigureLists();
    }

    void ApplyFonts()
    {
        SendMessageW(_title, WM_SETFONT, reinterpret_cast<WPARAM>(_titleFont), TRUE);
        for (HWND control : {_actionTitle, _summaryTitle, _resultsTitle})
        {
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(_sectionFont), TRUE);
        }
        for (HWND control : {_nameLabel, _nameEdit, _addButton, _searchButton, _reloadButton,
                             _customThresholdLabel, _customThresholdEdit,
                             _openCvThresholdLabel, _openCvThresholdEdit,
                             _algorithmLabel, _customCpuCheck, _customCudaCheck,
                             _openCvCpuCheck, _openCvCudaCheck})
        {
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(_font), TRUE);
        }
        for (HWND control : {_metricDb, _metricCuda, _statusText})
        {
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(_badgeFont), TRUE);
        }
        for (HWND control : {_subtitle, _hintText, _lastReport})
        {
            SendMessageW(control, WM_SETFONT, reinterpret_cast<WPARAM>(_smallFont), TRUE);
        }
        SendMessageW(_summaryList, WM_SETFONT, reinterpret_cast<WPARAM>(_font), TRUE);
        SendMessageW(_resultsList, WM_SETFONT, reinterpret_cast<WPARAM>(_font), TRUE);
    }

    void ConfigureLists()
    {
        ListView_SetExtendedListViewStyle(_summaryList, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);
        ListView_SetExtendedListViewStyle(_resultsList, LVS_EX_FULLROWSELECT | LVS_EX_GRIDLINES | LVS_EX_DOUBLEBUFFER);

        ListView_SetBkColor(_summaryList, kSurface);
        ListView_SetTextBkColor(_summaryList, kSurface);
        ListView_SetTextColor(_summaryList, kText);
        ListView_SetBkColor(_resultsList, kSurface);
        ListView_SetTextBkColor(_resultsList, kSurface);
        ListView_SetTextColor(_resultsList, kText);

        AddColumn(_summaryList, 0, 180, L"Алгоритм");
        AddColumn(_summaryList, 1, 78, L"Статус");
        AddColumn(_summaryList, 2, 92, L"Всего, мс");
        AddColumn(_summaryList, 3, 96, L"мс/лицо");
        AddColumn(_summaryList, 4, 72, L"Лиц");
        AddColumn(_summaryList, 5, 82, L"Пропуск");
        AddColumn(_summaryList, 6, 72, L"Отказ");
        AddColumn(_summaryList, 7, 76, L"Успех");
        AddColumn(_summaryList, 8, 76, L"Ошибки");
        AddColumn(_summaryList, 9, 90, L"Точность");
        AddColumn(_summaryList, 10, 92, L"Ложн. проп.");
        AddColumn(_summaryList, 11, 96, L"Ложн. отказ");
        AddColumn(_summaryList, 12, 92, L"Не то лицо");

        AddColumn(_resultsList, 0, 210, L"Файл");
        AddColumn(_resultsList, 1, 140, L"Ожидалось");
        AddColumn(_resultsList, 2, 72, L"В базе");
        AddColumn(_resultsList, 3, 180, L"Алгоритм");
        AddColumn(_resultsList, 4, 88, L"Решение");
        AddColumn(_resultsList, 5, 150, L"Найдено");
        AddColumn(_resultsList, 6, 78, L"Score");
        AddColumn(_resultsList, 7, 92, L"Время, мс");
        AddColumn(_resultsList, 8, 84, L"Итог");
        AddColumn(_resultsList, 9, 144, L"Тип ошибки");
        AddColumn(_resultsList, 10, 76, L"Порог");
    }

    bool IsChecked(HWND checkbox) const
    {
        return SendMessageW(checkbox, BM_GETCHECK, 0, 0) == BST_CHECKED;
    }

    std::vector<AlgorithmId> SelectedAlgorithmsFromUi() const
    {
        std::vector<AlgorithmId> selected;
        if (IsChecked(_customCpuCheck))
        {
            selected.push_back(AlgorithmId::CustomCpu);
        }
        if (IsChecked(_customCudaCheck))
        {
            selected.push_back(AlgorithmId::CustomCuda);
        }
        if (IsChecked(_openCvCpuCheck))
        {
            selected.push_back(AlgorithmId::OpenCvCpu);
        }
        if (IsChecked(_openCvCudaCheck))
        {
            selected.push_back(AlgorithmId::OpenCvCuda);
        }
        return selected;
    }

    void LayoutControls(int width, int height)
    {
        const int margin = 24;
        const int gap = 16;
        const int contentWidth = std::max(760, width - 2 * margin);

        MoveWindow(_title, margin, 18, 420, 36, TRUE);
        MoveWindow(_subtitle, margin, 56, std::max(420, width - 520), 24, TRUE);
        MoveWindow(_statusText, width - margin - 144, 22, 144, 32, TRUE);
        MoveWindow(_metricDb, width - margin - 320, 62, 150, 52, TRUE);
        MoveWindow(_metricCuda, width - margin - 156, 62, 132, 52, TRUE);

        int y = 126;
        const int actionHeight = 212;
        MoveWindow(_actionPanel, margin, y, contentWidth, actionHeight, TRUE);
        MoveWindow(_actionTitle, margin + 18, y + 14, 240, 24, TRUE);

        const int row1Y = y + 52;
        MoveWindow(_nameLabel, margin + 18, row1Y, 160, 22, TRUE);
        MoveWindow(_nameEdit, margin + 18, row1Y + 26, 280, 32, TRUE);
        MoveWindow(_addButton, margin + 318, row1Y + 25, 148, 34, TRUE);
        MoveWindow(_searchButton, margin + 480, row1Y + 25, 164, 34, TRUE);
        MoveWindow(_reloadButton, margin + 658, row1Y + 25, 152, 34, TRUE);

        const int row2LabelY = y + 114;
        const int row2EditY = row2LabelY + 24;
        MoveWindow(_customThresholdLabel, margin + 18, row2LabelY, 170, 22, TRUE);
        MoveWindow(_customThresholdEdit, margin + 18, row2EditY, 92, 30, TRUE);
        MoveWindow(_openCvThresholdLabel, margin + 140, row2LabelY, 190, 22, TRUE);
        MoveWindow(_openCvThresholdEdit, margin + 140, row2EditY, 92, 30, TRUE);
        MoveWindow(_hintText, margin + 258, row2EditY + 4, std::max(240, contentWidth - 510), 22, TRUE);
        MoveWindow(_lastReport, width - margin - 360, row2EditY + 4, 342, 22, TRUE);

        const int row3Y = y + 176;
        MoveWindow(_algorithmLabel, margin + 18, row3Y, 176, 24, TRUE);
        MoveWindow(_customCpuCheck, margin + 204, row3Y, 134, 24, TRUE);
        MoveWindow(_customCudaCheck, margin + 352, row3Y, 148, 24, TRUE);
        MoveWindow(_openCvCpuCheck, margin + 514, row3Y, 150, 24, TRUE);
        MoveWindow(_openCvCudaCheck, margin + 678, row3Y, 164, 24, TRUE);

        y += actionHeight + gap;
        const int summaryHeight = 194;
        MoveWindow(_summaryPanel, margin, y, contentWidth, summaryHeight, TRUE);
        MoveWindow(_summaryTitle, margin + 18, y + 14, 280, 24, TRUE);
        MoveWindow(_summaryList, margin + 18, y + 46, contentWidth - 36, summaryHeight - 62, TRUE);

        y += summaryHeight + gap;
        const int resultsHeight = std::max(220, height - y - margin);
        MoveWindow(_resultsPanel, margin, y, contentWidth, resultsHeight, TRUE);
        MoveWindow(_resultsTitle, margin + 18, y + 14, 260, 24, TRUE);
        MoveWindow(_resultsList, margin + 18, y + 46, contentWidth - 36, resultsHeight - 62, TRUE);
    }

    void HandleCommand(int id)
    {
        if (_busy)
        {
            return;
        }

        if (id == kAddButtonId)
        {
            AddFace();
        }
        else if (id == kSearchButtonId)
        {
            SearchFolder();
        }
        else if (id == kReloadButtonId)
        {
            _engine->ReloadDatabase();
            RefreshMetrics();
            PostLog("База обновлена: " + std::to_string(_engine->KnownFacesCount()) + " лиц.");
        }
    }

    void AddFace()
    {
        std::wstring wideName = GetWindowTextString(_nameEdit);
        if (wideName.empty())
        {
            MessageBoxW(_window, L"Введите имя лица перед добавлением.", L"Добавление лица", MB_OK | MB_ICONINFORMATION);
            return;
        }

        std::vector<std::filesystem::path> images = SelectImages(_window);
        if (images.empty())
        {
            return;
        }

        std::string name = WideToUtf8(wideName);
        PostMessageW(_window, kBusyMessage, TRUE, 0);
        PostStatus("Добавление лица...");
        PostLog("Добавление лица '" + name + "': выбрано " + std::to_string(images.size()) + " изображений.");

        std::thread([this, name, images = std::move(images)]() {
            try
            {
                _engine->EnrollFace(name, images, [this](const std::string& message) {
                    PostLog(message);
                });
                PostLog("Лицо сохранено в базе.");
            }
            catch (const std::exception& error)
            {
                PostLog(std::string("Ошибка добавления: ") + error.what());
            }

            PostMessageW(_window, kBusyMessage, FALSE, 0);
            PostStatus("Готово");
        }).detach();
    }

    void SearchFolder()
    {
        std::filesystem::path folder = SelectFolder(_window);
        if (folder.empty())
        {
            return;
        }

        RecognitionThresholds thresholds;
        thresholds.custom = ParseDoubleFromEdit(_customThresholdEdit, thresholds.custom);
        thresholds.openCv = ParseDoubleFromEdit(_openCvThresholdEdit, thresholds.openCv);
        std::vector<AlgorithmId> selectedAlgorithms = SelectedAlgorithmsFromUi();
        if (selectedAlgorithms.empty())
        {
            MessageBoxW(_window, L"Выберите хотя бы один алгоритм для проверки.", L"Проверка папки", MB_OK | MB_ICONINFORMATION);
            return;
        }

        ListView_DeleteAllItems(_resultsList);
        SeedAlgorithmTable("в работе", selectedAlgorithms);
        PostMessageW(_window, kBusyMessage, TRUE, 0);
        PostStatus("Проверка папки...");
        PostLog("Проверка папки: " + folder.string());
        PostLog("Пороги: моя LBPH = " + FormatDouble(thresholds.custom, 4) +
                ", OpenCV LBPH = " + FormatDouble(thresholds.openCv, 4));

        std::thread([this, folder = std::move(folder), thresholds, selectedAlgorithms = std::move(selectedAlgorithms)]() {
            try
            {
                SearchReport report = _engine->SearchDirectory(
                    folder,
                    [this](const std::string& message) {
                        PostLog(message);
                    },
                    thresholds,
                    selectedAlgorithms
                );

                std::filesystem::path csvPath =
                    std::filesystem::path(SRC_DIR) / ".." / "result" / "research_reports" /
                    ("face_control_" + TimestampForFile() + ".csv");
                _engine->SaveReportCsv(report, csvPath);

                auto* payload = new ReportPayload{std::move(report), std::move(csvPath)};
                PostMessageW(_window, kReportMessage, 0, reinterpret_cast<LPARAM>(payload));
                PostLog("Проверка завершена.");
            }
            catch (const std::exception& error)
            {
                PostLog(std::string("Ошибка проверки: ") + error.what());
            }

            PostMessageW(_window, kBusyMessage, FALSE, 0);
            PostStatus("Готово");
        }).detach();
    }

    void RenderReport(const ReportPayload& payload)
    {
        ListView_DeleteAllItems(_summaryList);
        ListView_DeleteAllItems(_resultsList);

        for (const AlgorithmRunSummary& summary : payload.report.summaries)
        {
            double accuracy = BalancedAccuracyPercent(summary);

            AddRow(_summaryList, {
                Utf8ToWide(AlgorithmTitle(summary.algorithm)),
                L"готово",
                Utf8ToWide(FormatDouble(summary.milliseconds)),
                Utf8ToWide(FormatDouble(summary.averageMilliseconds)),
                Utf8ToWide(std::to_string(summary.processed)),
                Utf8ToWide(std::to_string(summary.passed)),
                Utf8ToWide(std::to_string(summary.denied)),
                Utf8ToWide(std::to_string(summary.correct)),
                Utf8ToWide(std::to_string(summary.errors)),
                Utf8ToWide(FormatDouble(accuracy, 1) + "%"),
                Utf8ToWide(std::to_string(summary.falseAccepts)),
                Utf8ToWide(std::to_string(summary.falseRejects)),
                Utf8ToWide(std::to_string(summary.wrongMatches))
            });
        }

        for (const ImageAlgorithmResult& result : payload.report.results)
        {
            AddRow(_resultsList, {
                Utf8ToWide(result.imagePath.filename().string()),
                Utf8ToWide(result.expectedName),
                Utf8ToWide(result.expectedInDatabase ? "да" : "нет"),
                Utf8ToWide(AlgorithmTitle(result.algorithm)),
                Utf8ToWide(VerdictText(result.recognized)),
                Utf8ToWide(result.match.found ? result.match.name : "-"),
                Utf8ToWide(FormatDouble(result.match.similarity, 4)),
                Utf8ToWide(FormatDouble(result.totalMilliseconds)),
                Utf8ToWide(OutcomeText(result)),
                Utf8ToWide(ErrorText(result)),
                Utf8ToWide(FormatDouble(result.threshold, 4))
            });
        }

        SetText(_lastReport, "CSV-отчет: " + payload.csvPath.filename().string());
        PostLog("CSV-отчет сохранен: " + payload.csvPath.string());
    }

    void SeedAlgorithmTable(const std::string& state, const std::vector<AlgorithmId>& selectedAlgorithms)
    {
        ListView_DeleteAllItems(_summaryList);
        for (const AlgorithmInfo& algorithm : Algorithms())
        {
            if (!ContainsAlgorithm(selectedAlgorithms, algorithm.id))
            {
                continue;
            }
            AddRow(_summaryList, {
                Utf8ToWide(algorithm.title),
                Utf8ToWide(algorithm.usesCuda && !IsCudaAvailable() ? "нет CUDA" : state),
                L"-", L"-", L"-", L"-", L"-", L"-", L"-", L"-", L"-", L"-", L"-"
            });
        }
    }

    void RefreshMetrics()
    {
        SetText(_metricDb, "Лиц в базе\n" + std::to_string(_engine->KnownFacesCount()));
        SetText(_metricCuda, IsCudaAvailable() ? "CUDA\nдоступна" : "CUDA\nнет");
    }

    void SetBusy(bool busy)
    {
        _busy = busy;
        EnableWindow(_addButton, !busy);
        EnableWindow(_searchButton, !busy);
        EnableWindow(_reloadButton, !busy);
        EnableWindow(_nameEdit, !busy);
        EnableWindow(_customThresholdEdit, !busy);
        EnableWindow(_openCvThresholdEdit, !busy);
        EnableWindow(_customCpuCheck, !busy);
        EnableWindow(_openCvCpuCheck, !busy);
        EnableWindow(_customCudaCheck, !busy && IsCudaAvailable());
        EnableWindow(_openCvCudaCheck, !busy && IsCudaAvailable());

        if (!busy)
        {
            RefreshMetrics();
        }
    }

    void PostLog(const std::string& message)
    {
        std::cout << message << std::endl;
        OutputDebugStringW(Utf8ToWide(message + "\n").c_str());
    }

    void PostStatus(const std::string& message)
    {
        auto* wide = new std::wstring(Utf8ToWide(message));
        PostMessageW(_window, kStatusMessage, 0, reinterpret_cast<LPARAM>(wide));
    }

    HWND _window = nullptr;
    HWND _title = nullptr;
    HWND _subtitle = nullptr;
    HWND _statusText = nullptr;
    HWND _metricDb = nullptr;
    HWND _metricCuda = nullptr;
    HWND _actionPanel = nullptr;
    HWND _actionTitle = nullptr;
    HWND _nameLabel = nullptr;
    HWND _nameEdit = nullptr;
    HWND _addButton = nullptr;
    HWND _searchButton = nullptr;
    HWND _reloadButton = nullptr;
    HWND _customThresholdLabel = nullptr;
    HWND _customThresholdEdit = nullptr;
    HWND _openCvThresholdLabel = nullptr;
    HWND _openCvThresholdEdit = nullptr;
    HWND _algorithmLabel = nullptr;
    HWND _customCpuCheck = nullptr;
    HWND _customCudaCheck = nullptr;
    HWND _openCvCpuCheck = nullptr;
    HWND _openCvCudaCheck = nullptr;
    HWND _hintText = nullptr;
    HWND _lastReport = nullptr;
    HWND _summaryPanel = nullptr;
    HWND _summaryTitle = nullptr;
    HWND _summaryList = nullptr;
    HWND _resultsPanel = nullptr;
    HWND _resultsTitle = nullptr;
    HWND _resultsList = nullptr;
    HFONT _titleFont = nullptr;
    HFONT _sectionFont = nullptr;
    HFONT _font = nullptr;
    HFONT _smallFont = nullptr;
    HFONT _badgeFont = nullptr;
    HBRUSH _backgroundBrush = nullptr;
    HBRUSH _surfaceBrush = nullptr;
    HBRUSH _accentBrush = nullptr;
    bool _busy = false;
    std::unique_ptr<RecognitionEngine> _engine;
};
} // namespace

int RunGuiApp(const std::filesystem::path& modelPath, const std::filesystem::path& databaseRoot)
{
    CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED);
    MainWindow window(modelPath, databaseRoot);
    int result = window.Run();
    CoUninitialize();
    return result;
}
} // namespace biometrics
