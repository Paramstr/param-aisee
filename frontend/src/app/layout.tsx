import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Header } from "@/components/Header";
import "./globals.css";

const geistSans = Geist({
  variable: "--font-geist-sans",
  subsets: ["latin"],
});

const geistMono = Geist_Mono({
  variable: "--font-geist-mono",
  subsets: ["latin"],
});

export const metadata: Metadata = {
  title: "Osmo - Aisee",
  description: "Osmo Personal Computing",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body
        className={`${geistSans.variable} ${geistMono.variable} antialiased`}
      >
        <div className="h-screen max-h-screen bg-gradient-to-br from-gray-850 via-gray-950 to-gray-800 flex flex-col overflow-hidden">
          <Header />
          <main className="flex-1 min-h-0 overflow-hidden">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
