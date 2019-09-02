namespace CNN.UI
{
    using System;

    using CNN.BL.Constants;
    using CNN.BL.Utils;

    /// <summary>
    /// Класс входной точки приложения.
    /// </summary>
    class Program
    {
        /// <summary>
        /// Точка входа приложения.
        /// </summary>
        static void Main(string[] args)
        {
            Console.ForegroundColor = ConsoleColor.Green;

            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE);
            Console.ReadKey();

            var path = @"C:\Лабы\Учёба (3 семестр)\Jock.CNN\CNN\CNN.BL\Resources\1.bmp";

            var converter = new ImageConverterUtil(path);
            var matrix = converter.ConvertImageToMatrix();

            foreach (var value in matrix)
                Console.WriteLine(value.ToString());

            Console.ReadKey();
        }
    }
}
