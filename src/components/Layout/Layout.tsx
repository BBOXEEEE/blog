import React, { useEffect } from "react";

import { useTheme } from "@/hooks";

import * as styles from "./Layout.module.scss";

import { defineCustomElements as deckDeckGoHighlightElement } from '@deckdeckgo/highlight-code/dist/loader';

interface Props {
  children: React.ReactNode;
}

const Layout: React.FC<Props> = ({ children }: Props) => {
  const [{ mode }] = useTheme();

  useEffect(() => {
    document.documentElement.className = mode;
  }, [mode]);

  deckDeckGoHighlightElement();
  return <div className={styles.layout}>{children}</div>;
};

export default Layout;
