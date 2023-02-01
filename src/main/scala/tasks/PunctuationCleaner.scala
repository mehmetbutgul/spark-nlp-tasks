package tasks

import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp._
import org.apache.spark.ml.param.StringArrayParam
import org.apache.spark.ml.util.Identifiable
    /** This annotator takes a sequence of strings
     * (e.g. the output of a Tokenizer, Normalizer, Lemmatizer, and Stemmer)
     * and drops all the punctuations from the input sequences.
     * By default, it uses !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~ punctuations.
     * Punctuations can also be defined by explicitly setting them with setPunctuations(value: Array[String])
     *
     *  Example
     *  import spark.implicits._
     *  val documentAssembler = new DocumentAssembler()
     *    .setInputCol("text")
     *    .setOutputCol("document")
     *
     *  val tokenizer = new Tokenizer()
     *   .setInputCols(Array("document"))
     *   .setOutputCol("token")
     *
     *  val punctuationCleaner = new PunctuationCleaner()
     *   .setInputCols("token")
     *   .setOutputCol("cleanedPunctuation")
     *   .setExceptions(Array("?"))
     *   .addExceptions(Array("!"))
     *
     *  val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, punctuationCleaner))
     *  val df = Seq("Here are 14 common punctuation marks in English. " +
     *   "1. The Full Stop . " +
     *   "2. The Question Mark ? " +
     *   "3. Quotation Marks/Speech Marks \" " +
     *   "4. The Apostrophe ' " +
     *   "5. The Comma , " +
     *   "6. The Hyphen - " +
     *   "7. The dash - " +
     *   "8. The Exclamation Mark ! " +
     *   "9. The Colon  : " +
     *   "10. The Semicolon  ;  " +
     *   "11. Parentheses ( ) " +
     *   "12. Brackets [ ] " +
     *   "13. Ellipsis … " +
     *   "14. The Slash /")
     *   .toDF("text")
     * val result = pipeline.fit(df).transform(df)
     * result.selectExpr("cleanedPunctuation.result").show(false)
     *  +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
     * |result                                                                                                                                                                                                                                                                                                                                      |
     * +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
     * |[Here, are, 14, common, punctuation, marks, in, English, 1, The, Full, Stop, 2, The, Question, Mark, ?, 3, Quotation, Marks/Speech, Marks, 4, The, Apostrophe, 5, The, Comma, 6, The, Hyphen, 7, The, dash, 8, The, Exclamation, Mark, !, 9, The, Colon, 10, The, Semicolon, 11, Parentheses, 12, Brackets, 13, Ellipsis, …, 14, The, Slash]|* +--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
     */

  class PunctuationCleaner(override val uid: String)
    extends AnnotatorModel[PunctuationCleaner]
    with HasSimpleAnnotate[PunctuationCleaner] {

    /** Output annotator type: TOKEN */
  override val outputAnnotatorType: AnnotatorType = TOKEN

    /** Input annotator type: TOKEN */
  override val inputAnnotatorTypes: Array[AnnotatorType] = Array(TOKEN)
    /** overload constructor */
  def this() = this(Identifiable.randomUID("PUNCTUATION_CLEANER2"))

    /** The punctuations to be filtered out. By default english:  !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    * They are same as  \p{Punct} regex */
  val punctuations : StringArrayParam = new StringArrayParam(this, "punctuations", "The punctuations to be filtered out. By default it's english:  !\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~")

    /**The punctuations to be filtered out */
  def setPunctuations(values: Array[String]): this.type = {
    validatePunctuations(values)
    set(punctuations, values)
  }

    /** All elements in punctuations must have length == 1 */
  private def validatePunctuations(values: Array[String]): Unit = {
    require(
      values.forall(punctuation => punctuation.length == 1 || (punctuation.length == 2 && punctuation.substring(0, 1) == "\\")),
      "All elements in punctuations must have length == 1")
  }

    /** The punctuations to be filtered out */
  def getPunctuations(): Array[String] = $(punctuations)

      /** Punctuations that won't be affected by deletion */
  val exceptions : StringArrayParam = new StringArrayParam(this, "exceptions", "Punctuations that won't be affected by deletion")

    /** Punctuations that won't be affected by deletion */
  def setExceptions(values: Array[String]): this.type = {
    validatePunctuations(values)
    set(exceptions, values)
  }

    /** Punctuations that won't be affected by deletion */
  def getExceptions(): Array[String] = $(exceptions)

    /** add punctuations that won't be affected by deletion */
  def addExceptions(values: Array[String]): this.type = {
    validatePunctuations(values)
    set(exceptions, get(exceptions).getOrElse(Array[String]()).++:(values))
  }

  setDefault(
    inputCols -> Array(TOKEN),
    outputCol -> "cleanedPunctuations",
    punctuations -> Array(".", ",", ";", ":", "!", "*", "-", "(", ")", "\"", "'", "?", "#", "$", "%", "&", "+", "/", "\\", "<", ">", "=","@", "[", "]", "^", "_", "{", "}", "|", "~", "`") ,
    exceptions -> Array[String]()
  )

  override def annotate(annotations: Seq[Annotation]): Seq[Annotation] = {
    val processedExceptions = $(punctuations).filter(x => !$(exceptions).contains(x))
    val cleanedPunctuations = annotations.filter(token => !processedExceptions.contains(token.result.trim))
    cleanedPunctuations.map { tokenAnnotation =>
      Annotation(
        outputAnnotatorType,
        tokenAnnotation.begin,
        tokenAnnotation.end,
        tokenAnnotation.result,
        tokenAnnotation.metadata)
    }
  }
}
trait ReadablePretrainedPunctuationCleanerModel
  extends ParamsAndFeaturesReadable[PunctuationCleaner]
    with HasPretrained[PunctuationCleaner] {
  override val defaultModelName: Some[String] = Some("punctuation_en")

    /** Java compliant-overrides */
  override def pretrained(): PunctuationCleaner = super.pretrained()
  override def pretrained(name: String): PunctuationCleaner = super.pretrained(name)
  override def pretrained(name: String, lang: String): PunctuationCleaner =
    super.pretrained(name, lang)
  override def pretrained(name: String, lang: String, remoteLoc: String): PunctuationCleaner =
    super.pretrained(name, lang, remoteLoc)
}

    /** This is the companion object of [[PunctuationCleaner]]. Please refer to that class for the
    * documentation.
    */
    object PunctuationCleaner
        extends ParamsAndFeaturesReadable[PunctuationCleaner]
        with ReadablePretrainedPunctuationCleanerModel
