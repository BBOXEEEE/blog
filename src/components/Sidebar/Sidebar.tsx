import React from "react";

import { useSiteMetadata } from "@/hooks";

import { Author } from "./Author";
import { Contacts } from "./Contacts";
import { Copyright } from "./Copyright";
import { Menu } from "./Menu";
import { Category } from "./Category";

import * as styles from "./Sidebar.module.scss";

type Props = {
  isIndex?: boolean;
};

const Sidebar = ({ isIndex }: Props) => {
  const { author, copyright, menu, category } = useSiteMetadata();

  return (
    <div className={styles.sidebar}>
      <div className={styles.inner}>
        <Author author={author} isIndex={isIndex} />
        <Contacts contacts={author.contacts} />
        <Menu menu={menu} />
        <Category category={category} />
        <Copyright copyright={copyright} />
      </div>
    </div>
  );
};

export default Sidebar;
