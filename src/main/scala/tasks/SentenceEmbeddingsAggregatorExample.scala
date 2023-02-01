package tasks

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.annotators.sbd.pragmatic.SentenceDetector
import com.johnsnowlabs.nlp.embeddings.{BertSentenceEmbeddings, RoBertaSentenceEmbeddings}
import org.apache.spark.ml.Pipeline

object SentenceEmbeddingsAggregatorExample extends App {
  val spark = SparkUtil.getSession()
  import spark.implicits._

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text").setOutputCol("document")

  val sentence = new SentenceDetector()
    .setInputCols("document").setOutputCol("sentence")

  val tokenizer = new Tokenizer()
    .setInputCols("sentence").setOutputCol("token")

  val bertEmbeddings = BertSentenceEmbeddings.pretrained("sent_small_bert_L2_128")
    .setInputCols("sentence")
    .setOutputCol("bert_sentence_embeddings")

  val roBertaSentenceEmbeddings = RoBertaSentenceEmbeddings.pretrained()
    .setInputCols("sentence")
    .setOutputCol("roBerta_sentence_embeddings")
    .setCaseSensitive(true)

  val average = new SentenceEmbeddingsAggregator()
    .setInputCols("bert_sentence_embeddings","roBerta_sentence_embeddings")
    .setOutputCol("average")

  val pipeline = new Pipeline().setStages(
    Array(documentAssembler, sentence, tokenizer, bertEmbeddings, roBertaSentenceEmbeddings, average))
  val data = Seq("This is a sentence. This is second sentence.").toDF("text")
  val result = pipeline.fit(data).transform(data)

  result.selectExpr("explode(bert_sentence_embeddings) as bert").show( false)
  result.selectExpr("explode(roBerta_sentence_embeddings) as deBerta").show( false)
  result.selectExpr("explode(average) as result").show( false)
}
