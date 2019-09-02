namespace CNN.UI
{
    using System;

    using CNN.BL.Constants;

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
            Console.WriteLine(ConsoleMessageConstants.HELLO_MESSAGE);
            Console.ReadKey();
        }
    }
}
