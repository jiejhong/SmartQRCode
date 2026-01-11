@if "%DEBUG%"=="" @echo off
setlocal

set DIRNAME=%~dp0
if "%DIRNAME%" == "" set DIRNAME=.
set APP_HOME=%DIRNAME%

set CLASSPATH=%APP_HOME%\gradle\wrapper\gradle-wrapper.jar

@rem Execute Gradle
if not "%JAVA_HOME%"=="" (
  "%JAVA_HOME%\bin\java.exe" -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
) else (
  java -classpath "%CLASSPATH%" org.gradle.wrapper.GradleWrapperMain %*
)
if %ERRORLEVEL% neq 0 exit /b %ERRORLEVEL%

endlocal
