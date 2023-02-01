import com.johnsnowlabs.nlp.Annotation
import com.johnsnowlabs.nlp.AnnotatorType.TOKEN
import com.johnsnowlabs.nlp.annotator._
import com.johnsnowlabs.nlp.base._
import com.johnsnowlabs.nlp.util.io.ResourceHelper
import org.apache.spark.ml.Pipeline
import org.apache.spark.sql.functions.size
import org.scalatest.Tag
import org.scalatest.flatspec.AnyFlatSpec
import tasks.PunctuationCleaner

class PunctuationCleanerTestSpec extends AnyFlatSpec {
  val documentAssembler: DocumentAssembler = new DocumentAssembler()
    .setInputCol("text")
    .setOutputCol("document")

  val tokenizer: Tokenizer = new Tokenizer()
    .setInputCols(Array("document"))
    .setOutputCol("token")

  "PunctuationCleaner" should "correctly remove punctuations from tokenizer's results" taggedAs Tag("TEST_PUNCTUATION_CLEANER") in {

    val testData = ResourceHelper.spark
      .createDataFrame(
        Seq(1 ->
          """What is a Punctuation Mark?
         Examples: full stops(.) and others ? ! : ; " ' """))
      .toDF("id", "text")

    val expectedWithoutPunctuations = Seq(
      Annotation(TOKEN, 0, 3, "What", Map("sentence" -> "0")),
      Annotation(TOKEN, 5, 6, "is", Map("sentence" -> "0")),
      Annotation(TOKEN, 8, 8, "a", Map("sentence" -> "0")),
      Annotation(TOKEN, 10, 20, "Punctuation", Map("sentence" -> "0")),
      Annotation(TOKEN, 22, 25, "Mark", Map("sentence" -> "0")),
      Annotation(TOKEN, 38, 45, "Examples", Map("sentence" -> "0")),
      Annotation(TOKEN, 48, 51, "full", Map("sentence" -> "0")),
      Annotation(TOKEN, 53, 57, "stops", Map("sentence" -> "0")),
      Annotation(TOKEN, 58, 60, "(.)", Map("sentence" -> "0")),
      Annotation(TOKEN, 62, 64, "and", Map("sentence" -> "0")),
      Annotation(TOKEN, 66, 71, "others", Map("sentence" -> "0")))

    val punctuationCleaner = new PunctuationCleaner()
      .setInputCols("token").setOutputCol("cleanPunctuations")

    val pipeline = new Pipeline()
      .setStages(Array(documentAssembler, tokenizer, punctuationCleaner))

    val pipelineDF = pipeline.fit(testData).transform(testData)

    pipelineDF.select(size(pipelineDF("token.result")).as("totalTokens")).show
    pipelineDF.select(size(pipelineDF("cleanPunctuations.result")).as("totalCleanedTokens")).show

    val tokensWithoutPunctuations = Annotation.collect(pipelineDF, "cleanPunctuations").flatten.toSeq

    assert(tokensWithoutPunctuations == expectedWithoutPunctuations)

  }
}

