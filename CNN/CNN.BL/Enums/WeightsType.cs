namespace CNN.BL.Enums
{
    /// <summary>
    /// Типы весов.
    /// </summary>
    public enum WeightsType
    {
        /// <summary>
        /// Веса выходного слоя.
        /// </summary>
        Output,

        /// <summary>
        /// Веса скрытого слоя.
        /// </summary>
        Hidden,

        /// <summary>
        /// Веса ядра фильтра.
        /// </summary>
        Core,

        /// <summary>
        /// Неопределённый тип.
        /// </summary>
        None
    }
}
