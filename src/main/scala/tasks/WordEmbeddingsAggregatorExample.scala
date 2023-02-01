package tasks

import com.johnsnowlabs.nlp.DocumentAssembler
import com.johnsnowlabs.nlp.annotators.Tokenizer
import com.johnsnowlabs.nlp.embeddings.{AlbertEmbeddings, BertEmbeddings, DeBertaEmbeddings}
import org.apache.spark.ml.Pipeline

object WordEmbeddingsAggregatorExample extends App {
  val spark = SparkUtil.getSession()
  import spark.implicits._

  val documentAssembler = new DocumentAssembler()
    .setInputCol("text").setOutputCol("document")

  val tokenizer = new Tokenizer()
    .setInputCols("document").setOutputCol("token")

  val embeddings1 = BertEmbeddings.pretrained()
    .setInputCols("token", "document").setOutputCol("bert_embeddings")

  val embeddings2 = DeBertaEmbeddings.pretrained()
    .setInputCols("token", "document").setOutputCol("deBerta_embeddings")

  val embeddings3 = AlbertEmbeddings.pretrained()
    .setInputCols("token", "document").setOutputCol("albert_embeddings")

  val averager = new WordEmbeddingsAggregator()
    .setInputCols("bert_embeddings","deBerta_embeddings", "albert_embeddings")
    .setOutputCol("average")

  val pipeline = new Pipeline().setStages(
    Array(documentAssembler, tokenizer, embeddings1, embeddings2,embeddings3, averager))
  val data = Seq("This is a sentence.").toDF("text")
  val result = pipeline.fit(data).transform(data)

  result.selectExpr("explode(bert_embeddings) as bert").show( false)
  result.selectExpr("explode(deBerta_embeddings) as deBerta").show( false)
  result.selectExpr("explode(albert_embeddings) as albert").show(false)
  result.selectExpr("explode(average) as result").show( false)
}
