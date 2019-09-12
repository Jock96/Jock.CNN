namespace CNN.BL.Helpers
{
    using CNN.BL.Constants;

    using System;
    using System.Collections.Generic;
    using System.Linq;

    /// <summary>
    /// Класс-помощник для вывода ошибок.
    /// </summary>
    public static class ErrorHelper
    {
        /// <summary>
        /// Действия при ошибке получения весов.
        /// </summary>
        public static void GetDataError()
        {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                $"не удалось полуить значения весов фильтра!");

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
            Console.ReadKey();

            Environment.Exit(0);
        }

        /// <summary>
        /// Проверить файлы.
        /// </summary>
        /// <param name="files">Полученные файлы.</param>
        public static void CheckFiles(List<string> files)
        {
            if (!files.Any())
            {
                Console.BackgroundColor = ConsoleColor.Red;
                Console.ForegroundColor = ConsoleColor.Black;

                Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                    $"не удалось найти файлы весов!");

                Console.BackgroundColor = ConsoleColor.Black;
                Console.ForegroundColor = ConsoleColor.Green;

                Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
                Console.ReadKey();

                Environment.Exit(0);
            }
        }

        /// <summary>
        /// Действия при ошибке парсинга
        /// </summary>
        public static void ParseError()
        {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                $"не удалось преобразовать значение веса!");

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
            Console.ReadKey();

            Environment.Exit(0);
        }

        /// <summary>
        /// Проерка директории.
        /// </summary>
        public static void DirectoryError()
        {
            Console.BackgroundColor = ConsoleColor.Red;
            Console.ForegroundColor = ConsoleColor.Black;

            Console.WriteLine($"{ConsoleMessageConstants.ERROR_MESSAGE} " +
                $"указанная директория не существует!");

            Console.BackgroundColor = ConsoleColor.Black;
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.PRESS_ANY_KEY_MESSAGE);
            Console.ReadKey();

            Environment.Exit(0);
        }
    }
}
