import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'

export default function Home() {
  return (
    <div className={styles.container}>
      <Head>
        <title>Motionful API</title>
        <meta name="description" content="Motionful Project API" />
        <link rel="icon" href="/favicon.ico" />
      </Head>
    </div>
  )
}
