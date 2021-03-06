﻿namespace CNN.BL.Constants
{
    /// <summary>
    /// Класс констант сообщений консоли.
    /// </summary>
    public static class ConsoleMessageConstants
    {
        /// <summary>
        /// Приветственное сообщение.
        /// </summary>
        public const string HELLO_MESSAGE = "Вас приветствует демонстрационное " +
            "приложение свёрточной нейронной сети.";

        /// <summary>
        /// Сообщение доступных изображений.
        /// </summary>
        public const string IMAGES_IN_DIRECTORY_MESSAGE = "Доступные изображения в директории: ";

        /// <summary>
        /// Сообщение распознавания другого изображения.
        /// </summary>
        public const string RECOGNIZE_ANOTHER_MESSAGE = "Вы хотите распознать другое изображение? (Y/N)";

        /// <summary>
        /// Сообщение о выборе обучения.
        /// </summary>
        public const string LEARN_RESULT_MESSAGE = "Вы выбрали обучение.";

        /// <summary>
        /// Сообщение о выборе распознавания.
        /// </summary>
        public const string RECOGNIZE_RESULT_MESSAGE = "Вы выбрали распознавание.";

        /// <summary>
        /// Сообщение сохранения весов.
        /// </summary>
        public const string SAVE_MESSAGE = "Вы хотите сохранить текущие веса?(Y/N)";

        /// <summary>
        /// Сообщения выбора действия.
        /// </summary>
        public const string WORK_CHOISE_MESSAGE = "Для распознавания нажмите клавишу (R), " +
            "для обучениея клавишу (L).";

        /// <summary>
        /// Сообзение пути загрузки файлов.
        /// </summary>
        public const string LOAD_PATH_MESSAGE = "Введите путь до сохраненных файлов" +
            "(клавиша Enter для пути по умолчанию): ";

        /// <summary>
        /// Сообщение пути сохранения файлов.
        /// </summary>
        public const string SAVE_PATH_MESSAGE = "Введите путь для сохранения файлов" +
            "(клавиша Enter для пути по умолчанию): ";

        /// <summary>
        /// Сообщение для нажатия любой клавиши.
        /// </summary>
        public const string PRESS_ANY_KEY_MESSAGE = "Пожалуйста, нажмите любую клавишу.";

        /// <summary>
        /// Сообщение ввода имени файла.
        /// </summary>
        public const string PATH_MESSAGE = "Введите путь до файлов с изображениями " +
            "(клавиша Enter для пути по умолчанию): ";

        /// <summary>
        /// Сообщение выбора файла изображения.
        /// </summary>
        public const string FILE_NAME_MESSAGE = "Введите имя файла изображения: ";

        /// <summary>
        /// Сообщение ошибки.
        /// </summary>
        public const string ERROR_MESSAGE = "Ошибка: ";

        /// <summary>
        /// Сообщение информации.
        /// </summary>
        public const string INFO_MESSAGE = "Инфомрация: ";

        /// <summary>
        /// Сообщения выхода.
        /// </summary>
        public const string EXIT_MESSAGE = "Завершение работы...";
    }
}
