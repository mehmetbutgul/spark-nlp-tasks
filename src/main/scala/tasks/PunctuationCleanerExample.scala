package tasks

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotator.Tokenizer
import org.apache.spark.ml.Pipeline

object PunctuationCleanerExample extends App {
  val spark = SparkUtil.getSession()
  import spark.implicits._
  val documentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  val punctuationCleaner = new PunctuationCleaner()
    .setInputCols("token")
    .setOutputCol("cleanedPunctuation")
    .setExceptions(Array("?"))
    .addExceptions(Array("!"))

  val pipeline = new Pipeline().setStages(Array(documentAssembler, tokenizer, punctuationCleaner))
  val df = Seq("Here are 14 common punctuation marks in English. " +
    "1. The Full Stop . " +
    "2. The Question Mark ? " +
    "3. Quotation Marks/Speech Marks \" " +
    "4. The Apostrophe ' " +
    "5. The Comma , " +
    "6. The Hyphen - " +
    "7. The dash - " +
    "8. The Exclamation Mark ! " +
    "9. The Colon  : " +
    "10. The Semicolon  ;  " +
    "11. Parentheses ( ) " +
    "12. Brackets [ ] " +
    "13. Ellipsis … " +
    "14. The Slash /")
    .toDF("text")
  val result = pipeline.fit(df).transform(df)
  result.selectExpr("cleanedPunctuation.result").show(false)
}
