/*
 * Substituto vazio do plugin de localização do cliente Flet, usado só no pacote MSIX.
 *
 * O cliente Flet para Windows ("full") registra o plugin geolocator ao abrir, mesmo sem nenhum
 * controle de localização na página. Instalado como pacote (Microsoft Store), isso faz o Windows
 * pedir "Permitir que Sigra acesse sua localização precisa?" na primeira abertura, e o Sigra
 * não usa localização. Esta DLL tem o mesmo nome e a única função que o flet.exe importa dela
 * (GeolocatorWindowsRegisterWithRegistrar), e não registra nada.
 *
 * Compilada por scripts/gerar_msix.ps1 com o Zig (pacote ziglang do PyPI):
 *   zig cc -shared -O2 -target x86_64-windows-gnu geolocator_windows_plugin.c
 *          -o geolocator_windows_plugin.dll
 */
#include <windows.h>

__declspec(dllexport) void GeolocatorWindowsRegisterWithRegistrar(void *registrar) {
    (void)registrar;
}

BOOL WINAPI DllMain(HINSTANCE instancia, DWORD motivo, LPVOID reservado) {
    (void)instancia;
    (void)motivo;
    (void)reservado;
    return TRUE;
}
